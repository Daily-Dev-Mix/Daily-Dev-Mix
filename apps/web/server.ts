import crypto from "crypto";
import express, { NextFunction, Request, Response } from "express";
import path from "path";
import { fileURLToPath } from "url";
import SpotifyWebApi from "spotify-web-api-node";
import cookieParser from "cookie-parser";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);
const AUTH_STATE_COOKIE = "spotify_auth_state";
const AUTH_CLIENT_ORIGIN_COOKIE = "spotify_client_origin";
const DEFAULT_SPOTIFY_REDIRECT_URIS = [
  "https://ais-dev-2phkeougjzy46tmi7u2dnl-600786790133.us-east1.run.app/auth/callback",
  "https://ais-pre-2phkeougjzy46tmi7u2dnl-600786790133.us-east1.run.app/auth/callback",
  "http://127.0.0.1:3000/auth/callback",
];
const SPOTIFY_SCOPES = [
  "user-read-private",
  "user-read-email",
  "user-read-playback-state",
  "user-modify-playback-state",
  "user-read-currently-playing",
  "user-read-recently-played",
  "user-top-read",
  "playlist-modify-private",
];
const PLAYLIST_WRITE_SCOPES = [
  "playlist-modify-private",
];
const SPOTIFY_PLAYLIST_ITEMS_BATCH_SIZE = 100;

interface SimplifiedTrack {
  id: string;
  name: string;
  uri: string;
  artists: Array<{ name: string }>;
  album: { images: Array<{ url: string }> };
}

interface PlaylistSuggestion {
  name: string;
  description: string;
  reasoning: string;
  seedTracks: string[];
}

interface SpotifyCurrentUserProfile {
  id: string;
  display_name?: string | null;
}

interface AuthenticatedRequest extends Request {
  spotifyApi?: SpotifyWebApi;
  spotifyAccessToken?: string;
}

const memoryCache = new Map<string, { expiresAt: number; value: unknown }>();

function getAllowedRedirectUris(): Set<string> {
  const configuredUris = process.env.SPOTIFY_ALLOWED_REDIRECT_URIS
    ?.split(",")
    .map((value) => value.trim())
    .filter(Boolean);

  return new Set(configuredUris?.length ? configuredUris : DEFAULT_SPOTIFY_REDIRECT_URIS);
}

function parseSpotifyScopeString(scopeValue: unknown): string[] {
  if (typeof scopeValue !== "string") {
    return [];
  }

  return scopeValue
    .split(/\s+/)
    .map((value) => value.trim())
    .filter(Boolean);
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (typeof error === "object" && error !== null) {
    const maybeError = error as {
      body?:
        | string
        | {
            error?: string | { message?: string; reason?: string };
            error_description?: string;
            message?: string;
          };
      message?: string;
      statusCode?: number;
    };
    const errorBody = typeof maybeError.body === "object" && maybeError.body !== null ? maybeError.body : null;

    if (typeof maybeError.body === "string" && maybeError.body.trim()) {
      return maybeError.body;
    }

    if (typeof errorBody?.error === "string" && errorBody.error.trim()) {
      return errorBody.error_description ? `${errorBody.error} ${errorBody.error_description}`.trim() : errorBody.error;
    }

    if (typeof errorBody?.error === "object" && errorBody.error !== null && errorBody.error.message) {
      return errorBody.error.reason ? `${errorBody.error.message} ${errorBody.error.reason}`.trim() : errorBody.error.message;
    }

    if (errorBody?.message) {
      return errorBody.message;
    }

    if (maybeError.message) {
      return maybeError.message;
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
}

function isSpotifyForbiddenError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) {
    return false;
  }

  const maybeError = error as {
    statusCode?: number;
    body?: { error?: { status?: number } };
  };

  return maybeError.statusCode === 403 || maybeError.body?.error?.status === 403;
}

function createSpotifyApi(options: {
  redirectUri?: string;
  accessToken?: string;
  refreshToken?: string;
} = {}): SpotifyWebApi {
  const spotifyApi = new SpotifyWebApi({
    clientId: process.env.SPOTIFY_CLIENT_ID,
    clientSecret: process.env.SPOTIFY_CLIENT_SECRET,
    redirectUri: options.redirectUri,
  });

  if (options.accessToken) {
    spotifyApi.setAccessToken(options.accessToken);
  }

  if (options.refreshToken) {
    spotifyApi.setRefreshToken(options.refreshToken);
  }

  return spotifyApi;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function getTokenFingerprint(accessToken: string): string {
  return crypto.createHash("sha256").update(accessToken).digest("hex").slice(0, 16);
}

function readCache<T>(cacheKey: string): T | null {
  const cachedValue = memoryCache.get(cacheKey);

  if (!cachedValue) {
    return null;
  }

  if (cachedValue.expiresAt <= Date.now()) {
    memoryCache.delete(cacheKey);
    return null;
  }

  return cachedValue.value as T;
}

function writeCache<T>(cacheKey: string, value: T, ttlMs: number): T {
  memoryCache.set(cacheKey, {
    value,
    expiresAt: Date.now() + ttlMs,
  });

  return value;
}

function isSpotifyRateLimitError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) {
    return false;
  }

  const maybeError = error as {
    statusCode?: number;
    body?: { error?: { status?: number } };
  };

  return maybeError.statusCode === 429 || maybeError.body?.error?.status === 429;
}

function getSpotifyRetryAfterMs(error: unknown): number | null {
  if (typeof error !== "object" || error === null) {
    return null;
  }

  const maybeError = error as {
    headers?: Record<string, string | string[] | undefined>;
  };
  const retryAfterHeader = maybeError.headers?.["retry-after"];
  const retryAfterValue = Array.isArray(retryAfterHeader) ? retryAfterHeader[0] : retryAfterHeader;
  const retryAfterSeconds = Number.parseInt(retryAfterValue || "", 10);

  if (!Number.isFinite(retryAfterSeconds) || retryAfterSeconds < 0) {
    return null;
  }

  return retryAfterSeconds * 1000;
}

async function withSpotifyRetry<T>(operation: () => Promise<T>, maxAttempts = 4): Promise<T> {
  let delayMs = 600;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      if (!isSpotifyRateLimitError(error) || attempt === maxAttempts) {
        throw error;
      }

      const retryDelay = getSpotifyRetryAfterMs(error) ?? delayMs;
      await sleep(retryDelay);
      delayMs *= 2;
    }
  }

  throw new Error("Spotify request retry unexpectedly exhausted.");
}

function chunkItems<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];

  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }

  return chunks;
}

async function spotifyApiRequest<T>(path: string, accessToken: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`https://api.spotify.com/v1${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.headers || {}),
    },
  });

  if (response.ok) {
    if (response.status === 204) {
      return undefined as T;
    }

    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/json")
      ? ((await response.json()) as T)
      : ((await response.text()) as T);
  }

  const contentType = response.headers.get("content-type") || "";
  const responseBody = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : await response.text().catch(() => null);

  throw {
    statusCode: response.status,
    body: responseBody,
    headers: Object.fromEntries(response.headers.entries()),
    message:
      (typeof responseBody === "string" && responseBody.trim()) ||
      response.statusText ||
      `Spotify API request failed with status ${response.status}.`,
  };
}

async function getCurrentSpotifyUserProfile(accessToken: string): Promise<SpotifyCurrentUserProfile> {
  return spotifyApiRequest<SpotifyCurrentUserProfile>("/me", accessToken);
}

async function createSpotifyPlaylist(
  accessToken: string,
  payload: { name: string; description: string; public: boolean },
): Promise<{ id: string; name: string }> {
  return spotifyApiRequest<{ id: string; name: string }>("/me/playlists", accessToken, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function addItemsToSpotifyPlaylist(playlistId: string, itemUris: string[], accessToken: string): Promise<void> {
  const validUris = itemUris.filter((uri) => uri && !uri.startsWith("spotify:local:"));

  for (const uriBatch of chunkItems(validUris, SPOTIFY_PLAYLIST_ITEMS_BATCH_SIZE)) {
    await spotifyApiRequest<void>(`/playlists/${encodeURIComponent(playlistId)}/items`, accessToken, {
      method: "POST",
      body: JSON.stringify({ uris: uriBatch }),
    });
  }
}

function buildSpotifyWriteForbiddenMessage(error: unknown, currentUserId?: string): string {
  const baseMessage = getErrorMessage(
    error,
    `Spotify rejected playlist creation. Reconnect Spotify and grant ${PLAYLIST_WRITE_SCOPES.join(" and ")} access, then try again.`,
  );

  if (!/forbidden/i.test(baseMessage)) {
    return baseMessage;
  }

  const accountHint = currentUserId
    ? ` Authenticated Spotify account: ${currentUserId}.`
    : "";

  return `${baseMessage}${accountHint} If this app is in Development Mode, make sure that Spotify account is added in the Spotify Developer Dashboard under Users Management, then reconnect Spotify to refresh the token.`;
}

async function getCachedSpotifyValue<T>(
  cacheKey: string,
  ttlMs: number,
  fetcher: () => Promise<T>,
): Promise<T> {
  const cachedValue = readCache<T>(cacheKey);
  if (cachedValue !== null) {
    return cachedValue;
  }

  const freshValue = await withSpotifyRetry(fetcher);
  return writeCache(cacheKey, freshValue, ttlMs);
}

function parseUrl(url: string, label: string): URL {
  try {
    const parsedUrl = new URL(url);
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      throw new Error(`${label} must use http or https.`);
    }
    return parsedUrl;
  } catch {
    throw new Error(`${label} must be an absolute URL.`);
  }
}

function isLoopbackHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]" || hostname === "::1";
}

function resolveRequestOrigin(req: Request): string {
  const forwardedProto = req.get("x-forwarded-proto")?.split(",")[0]?.trim();
  const forwardedHost = req.get("x-forwarded-host")?.split(",")[0]?.trim();
  const protocol = forwardedProto || req.protocol;
  const host = forwardedHost || req.get("host");

  if (!host) {
    throw new Error("Unable to determine the current request host.");
  }

  return `${protocol}://${host}`;
}

function resolveSpotifyRedirectUri(req: Request): string {
  const explicitRedirectUri = process.env.SPOTIFY_REDIRECT_URI?.trim();
  const candidate = explicitRedirectUri || new URL("/auth/callback", resolveRequestOrigin(req)).toString();
  const parsedRedirectUri = parseUrl(candidate, "Spotify redirect URI");

  // Spotify requires explicit loopback IP literals for local HTTP redirects.
  // Normalize local callbacks to 127.0.0.1 so the allowlist and browser cookies stay consistent.
  if (!explicitRedirectUri && isLoopbackHost(parsedRedirectUri.hostname)) {
    parsedRedirectUri.hostname = "127.0.0.1";
  }

  if (parsedRedirectUri.protocol !== "https:" && !isLoopbackHost(parsedRedirectUri.hostname)) {
    throw new Error("Spotify redirect URI must use HTTPS unless the app is running on localhost.");
  }

  const normalizedRedirectUri = parsedRedirectUri.toString();
  const allowedRedirectUris = getAllowedRedirectUris();

  if (!allowedRedirectUris.has(normalizedRedirectUri)) {
    throw new Error(
      `Spotify redirect URI "${normalizedRedirectUri}" is not allowed. Add it to SPOTIFY_ALLOWED_REDIRECT_URIS and the Spotify dashboard.`,
    );
  }

  return normalizedRedirectUri;
}

function buildCookieOptions(req: Request) {
  const secure =
    req.secure ||
    req.get("x-forwarded-proto")
      ?.split(",")
      .some((value) => value.trim() === "https") ||
    false;

  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure,
    path: "/",
  };
}

function getAllowedFrontendOrigins(): Set<string> {
  const frontendOrigins = new Set<string>();

  for (const redirectUri of getAllowedRedirectUris()) {
    frontendOrigins.add(parseUrl(redirectUri, "Spotify redirect URI").origin);
  }

  return frontendOrigins;
}

function isAllowedFrontendOrigin(origin: string): boolean {
  const trimmedOrigin = origin.trim();
  if (!trimmedOrigin) {
    return false;
  }

  const parsedOrigin = parseUrl(trimmedOrigin, "Frontend origin");
  if (isLoopbackHost(parsedOrigin.hostname)) {
    return true;
  }

  return getAllowedFrontendOrigins().has(parsedOrigin.origin);
}

function resolveClientOrigin(req: Request): string | null {
  const requestOrigin = req.get("origin")?.trim();
  if (requestOrigin && isAllowedFrontendOrigin(requestOrigin)) {
    return requestOrigin;
  }

  const referer = req.get("referer")?.trim();
  if (!referer) {
    return null;
  }

  try {
    const refererOrigin = new URL(referer).origin;
    return isAllowedFrontendOrigin(refererOrigin) ? refererOrigin : null;
  } catch {
    return null;
  }
}

function applyCorsHeaders(req: Request, res: Response): void {
  const requestOrigin = req.get("origin")?.trim();
  if (!requestOrigin || !isAllowedFrontendOrigin(requestOrigin)) {
    return;
  }

  res.set("Access-Control-Allow-Origin", requestOrigin);
  res.set("Access-Control-Allow-Credentials", "true");
  res.set("Access-Control-Allow-Headers", "Authorization, Content-Type");
  res.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.append("Vary", "Origin");
}

function getBearerToken(req: Request): string | null {
  const authorizationHeader = req.get("authorization");
  if (!authorizationHeader?.startsWith("Bearer ")) {
    return null;
  }

  return authorizationHeader.slice("Bearer ".length).trim();
}

function normalizeTrack(track: unknown): SimplifiedTrack {
  const value = (track ?? {}) as {
    id?: string;
    name?: string;
    uri?: string;
    artists?: Array<{ name?: string }>;
    album?: { images?: Array<{ url?: string }> };
  };

  return {
    id: value.id || value.uri || crypto.randomUUID(),
    name: value.name || "Unknown track",
    uri: value.uri || "",
    artists: Array.isArray(value.artists) && value.artists.length
      ? value.artists.map((artist) => ({ name: artist?.name || "Unknown artist" }))
      : [{ name: "Unknown artist" }],
    album: {
      images: Array.isArray(value.album?.images)
        ? value.album.images
            .map((image) => ({ url: image?.url || "" }))
            .filter((image) => Boolean(image.url))
        : [],
    },
  };
}

function mergeTracks(primary: SimplifiedTrack[], secondary: SimplifiedTrack[]): SimplifiedTrack[] {
  const seen = new Set<string>();
  const merged: SimplifiedTrack[] = [];

  for (const track of [...primary, ...secondary]) {
    const key = track.id || track.uri;
    if (!key || seen.has(key)) {
      continue;
    }

    seen.add(key);
    merged.push(track);
  }

  return merged;
}

function parseSuggestionResponse(rawText: string): PlaylistSuggestion {
  if (!rawText.trim()) {
    throw new Error("Gemini returned an empty response.");
  }

  const cleanedJson = rawText.replace(/```json\n?|```/g, "").trim();
  const parsed = JSON.parse(cleanedJson) as Partial<PlaylistSuggestion>;

  if (!parsed.name || !parsed.description || !parsed.reasoning || !Array.isArray(parsed.seedTracks)) {
    throw new Error("Gemini returned an incomplete playlist suggestion.");
  }

  return {
    name: parsed.name,
    description: parsed.description,
    reasoning: parsed.reasoning,
    seedTracks: parsed.seedTracks.map((track) => String(track)),
  };
}

function renderCallbackPage(payload: Record<string, unknown>, statusMessage: string, targetOrigin: string): string {
  const serializedPayload = JSON.stringify(payload).replace(/</g, "\\u003c");
  const serializedTargetOrigin = JSON.stringify(targetOrigin).replace(/</g, "\\u003c");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Spotify Authentication</title>
    <style>
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at top, rgba(30, 215, 96, 0.22), transparent 35%),
          linear-gradient(160deg, #100d09, #1d140e 45%, #261912);
        color: #f8f3ee;
        font: 16px/1.5 "Avenir Next", "Segoe UI", sans-serif;
      }
      .panel {
        max-width: 28rem;
        margin: 2rem;
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.06);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
        text-align: center;
      }
      .panel p {
        margin: 0;
      }
    </style>
  </head>
  <body>
    <div class="panel">
      <p>${statusMessage}</p>
    </div>
    <script>
      const payload = ${serializedPayload};
      const targetOrigin = ${serializedTargetOrigin};
      if (window.opener) {
        window.opener.postMessage(payload, targetOrigin);
        window.close();
      }
    </script>
  </body>
</html>`;
}

const app = express();
const publicDir = path.join(__dirname, "public");

app.set("trust proxy", true);
app.use(express.json({ limit: "1mb" }));
app.use(cookieParser());
app.use((req, res, next) => {
  applyCorsHeaders(req, res);

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  next();
});
app.use(express.static(publicDir));

const requireSpotifyAccessToken = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  const accessToken = getBearerToken(req);

  if (!accessToken) {
    return res.status(401).json({ error: "Missing Spotify access token." });
  }

  req.spotifyApi = createSpotifyApi({ accessToken });
  req.spotifyAccessToken = accessToken;
  next();
};

app.get("/api/auth/url", (req, res) => {
  try {
    const redirectUri = resolveSpotifyRedirectUri(req);
    const spotifyApi = createSpotifyApi({ redirectUri });
    const state = crypto.randomBytes(24).toString("hex");
    const clientOrigin = resolveClientOrigin(req) || resolveRequestOrigin(req);

    res.cookie(AUTH_STATE_COOKIE, state, {
      ...buildCookieOptions(req),
      maxAge: 10 * 60 * 1000,
    });
    res.cookie(AUTH_CLIENT_ORIGIN_COOKIE, clientOrigin, {
      ...buildCookieOptions(req),
      maxAge: 10 * 60 * 1000,
    });

    const authorizeUrl = spotifyApi.createAuthorizeURL(SPOTIFY_SCOPES, state, true);
    res.json({ url: authorizeUrl, redirectUri });
  } catch (error) {
    console.error("Failed to create Spotify auth URL:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to create Spotify authorization URL."),
    });
  }
});

app.get("/auth/callback", async (req, res) => {
  const { code, state } = req.query;
  const expectedState = req.cookies[AUTH_STATE_COOKIE] as string | undefined;
  const clientOriginCookie = req.cookies[AUTH_CLIENT_ORIGIN_COOKIE] as string | undefined;
  const callbackTargetOrigin =
    clientOriginCookie && isAllowedFrontendOrigin(clientOriginCookie) ? clientOriginCookie : resolveRequestOrigin(req);

  res.clearCookie(AUTH_STATE_COOKIE, buildCookieOptions(req));
  res.clearCookie(AUTH_CLIENT_ORIGIN_COOKIE, buildCookieOptions(req));
  res.set("Cache-Control", "no-store");

  if (!code || typeof code !== "string") {
    return res
      .status(400)
      .send(
        renderCallbackPage(
          { type: "SPOTIFY_AUTH_ERROR", message: "Spotify did not return an authorization code." },
          "Authentication failed. You can close this window.",
          callbackTargetOrigin,
        ),
      );
  }

  if (!state || typeof state !== "string" || !expectedState || state !== expectedState) {
    return res
      .status(400)
      .send(
        renderCallbackPage(
          { type: "SPOTIFY_AUTH_ERROR", message: "The Spotify login state did not match." },
          "Authentication failed. You can close this window.",
          callbackTargetOrigin,
        ),
      );
  }

  try {
    const redirectUri = resolveSpotifyRedirectUri(req);
    const spotifyApi = createSpotifyApi({ redirectUri });
    const data = await spotifyApi.authorizationCodeGrant(code);
    const { access_token, refresh_token, expires_in } = data.body;
    const grantedScopes = parseSpotifyScopeString((data.body as { scope?: string }).scope);

    res.redirect(
      `/dashboard.html?accessToken=${encodeURIComponent(access_token)}&refreshToken=${encodeURIComponent(refresh_token)}&expiresIn=${expires_in}&scopes=${encodeURIComponent(grantedScopes.join(" "))}`
    );
    
  } catch (error) {
    console.error("Spotify callback failed:", error);
    res
      .status(500)
      .send(
        renderCallbackPage(
          {
            type: "SPOTIFY_AUTH_ERROR",
            message: getErrorMessage(error, "Spotify authentication failed."),
          },
          "Authentication failed. You can close this window.",
          callbackTargetOrigin,
        ),
      );
  }
});

app.post("/api/auth/refresh", async (req, res) => {
  const refreshToken = typeof req.body.refreshToken === "string" ? req.body.refreshToken.trim() : "";

  if (!refreshToken) {
    return res.status(400).json({ error: "No refresh token provided." });
  }

  try {
    const spotifyApi = createSpotifyApi({ refreshToken });
    const data = await spotifyApi.refreshAccessToken();
    const refreshedScopes = parseSpotifyScopeString((data.body as { scope?: string }).scope);

    res.json({
      accessToken: data.body.access_token,
      refreshToken: data.body.refresh_token || refreshToken,
      expiresIn: data.body.expires_in,
      scopes: refreshedScopes,
    });
  } catch (error) {
    console.error("Failed to refresh Spotify token:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to refresh Spotify token."),
    });
  }
});

app.get("/api/spotify/me", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  try {
    const cacheKey = `${getTokenFingerprint(req.spotifyAccessToken!)}:me`;
    const profile = await getCachedSpotifyValue(cacheKey, 5 * 60 * 1000, async () => {
      const data = await req.spotifyApi!.getMe();
      return data.body;
    });
    res.json(profile);
  } catch (error) {
    console.error("Failed to fetch Spotify profile:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to fetch Spotify profile."),
    });
  }
});

app.get("/api/spotify/currently-playing", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  try {
    const data = await withSpotifyRetry(() => req.spotifyApi!.getMyCurrentPlayingTrack());
    res.json(data.body || null);
  } catch (error) {
    console.error("Failed to fetch current Spotify track:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to fetch the current Spotify track."),
    });
  }
});

app.get("/api/spotify/top-tracks", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  try {
    const requestedLimit = Number.parseInt(String(req.query.limit || "20"), 10);
    const limit = Number.isFinite(requestedLimit) ? Math.min(Math.max(requestedLimit, 1), 50) : 20;
    const cacheKey = `${getTokenFingerprint(req.spotifyAccessToken!)}:top-tracks:${limit}`;
    const topTracks = await getCachedSpotifyValue(cacheKey, 10 * 60 * 1000, async () => {
      const data = await req.spotifyApi!.getMyTopTracks({ limit });
      return data.body;
    });
    res.json(topTracks);
  } catch (error) {
    console.error("Failed to fetch top Spotify tracks:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to fetch Spotify top tracks."),
    });
  }
});

app.get("/api/spotify/recently-played", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  try {
    const requestedLimit = Number.parseInt(String(req.query.limit || "10"), 10);
    const limit = Number.isFinite(requestedLimit) ? Math.min(Math.max(requestedLimit, 1), 50) : 10;
    const cacheKey = `${getTokenFingerprint(req.spotifyAccessToken!)}:recently-played:${limit}`;
    const recentlyPlayed = await getCachedSpotifyValue(cacheKey, 60 * 1000, async () => {
      const data = await req.spotifyApi!.getMyRecentlyPlayedTracks({ limit });
      return data.body;
    });
    res.json(recentlyPlayed);
  } catch (error) {
    console.error("Failed to fetch recently played tracks:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to fetch Spotify recently played tracks."),
    });
  }
});

app.post("/api/spotify/create-playlist", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  const name = typeof req.body.name === "string" ? req.body.name.trim() : "";
  const description = typeof req.body.description === "string" ? req.body.description.trim() : "";
  const trackUris = Array.isArray(req.body.trackUris)
    ? req.body.trackUris.map((trackUri: unknown) => String(trackUri)).filter(Boolean)
    : [];
  let currentUserId = "";

  if (!name) {
    return res.status(400).json({ error: "Playlist name is required." });
  }

  try {
    const currentUser = await withSpotifyRetry(() => getCurrentSpotifyUserProfile(req.spotifyAccessToken!));
    currentUserId = currentUser.id;
    const playlist = await withSpotifyRetry(() => createSpotifyPlaylist(req.spotifyAccessToken!, {
      name,
      description,
      public: false,
    }));

    if (trackUris.length > 0) {
      try {
        await withSpotifyRetry(() => addItemsToSpotifyPlaylist(playlist.id, trackUris, req.spotifyAccessToken!));
      } catch (error) {
        console.error("Failed to add items to Spotify playlist:", error);
        throw error;
      }
    }

    res.json(playlist);
  } catch (error) {
    console.error("Failed to create Spotify playlist:", error);

    if (isSpotifyForbiddenError(error)) {
      return res.status(403).json({
        error: buildSpotifyWriteForbiddenMessage(error, currentUserId),
      });
    }

    res.status(500).json({
      error: getErrorMessage(error, "Failed to create Spotify playlist."),
    });
  }
});

app.post("/api/ai/suggestion", requireSpotifyAccessToken, async (req: AuthenticatedRequest, res) => {
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "GEMINI_API_KEY is not configured on the server." });
  }

  const sessionType = typeof req.body.sessionType === "string" ? req.body.sessionType.trim() : "focus";
  const requestedExplorationLevel = Number(req.body.explorationLevel);
  const explorationLevel = Number.isFinite(requestedExplorationLevel)
    ? Math.min(Math.max(requestedExplorationLevel, 0), 100)
    : 50;
  const anchorTracks = Array.isArray(req.body.anchorTracks)
    ? req.body.anchorTracks.map((track: unknown) => String(track).trim()).filter(Boolean).slice(0, 5)
    : [];
  const submittedTracks = Array.isArray(req.body.tracks)
    ? req.body.tracks.map((track: unknown) => normalizeTrack(track))
    : [];

  let contextTracks = submittedTracks;

  if (contextTracks.length < 3) {
    try {
      const tokenFingerprint = getTokenFingerprint(req.spotifyAccessToken!);
      const [topTracksResponse, recentlyPlayedResponse] = await Promise.all([
        getCachedSpotifyValue(`${tokenFingerprint}:top-tracks:10`, 10 * 60 * 1000, async () => {
          const response = await req.spotifyApi!.getMyTopTracks({ limit: 10 });
          return response.body;
        }),
        getCachedSpotifyValue(`${tokenFingerprint}:recently-played:10`, 60 * 1000, async () => {
          const response = await req.spotifyApi!.getMyRecentlyPlayedTracks({ limit: 10 });
          return response.body;
        }),
      ]);
      const topTracks = topTracksResponse.items.map((track) => normalizeTrack(track));
      const recentTracks = recentlyPlayedResponse.items.map((item) => normalizeTrack(item.track));
      contextTracks = mergeTracks(contextTracks, mergeTracks(recentTracks, topTracks));
    } catch (error) {
      console.warn("Unable to supplement session with top tracks:", error);
    }
  }

  if (!contextTracks.length) {
    return res.status(400).json({
      error: "No session tracks were available to analyze.",
    });
  }

  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const model = process.env.GEMINI_MODEL || "gemini-3-flash-preview";
    const trackList = contextTracks
      .map((track) => `${track.name} by ${track.artists.map((artist) => artist.name).join(", ")}`)
      .join("\n");

    const response = await ai.models.generateContent({
      model,
      contents: `Analyze this music session for the activity: ${sessionType}.
Tracks listened to (supplemented with top tracks when the session is short):
${trackList}

Anchor tracks for this session type:
${anchorTracks.length ? anchorTracks.join("\n") : "No anchor tracks were supplied."}

Exploration Level: ${explorationLevel}% (0% means stay very close to the user's current taste, 100% means introduce much more discovery).

Create a personalized playlist suggestion for future ${sessionType} sessions.
Explain why the playlist fits this session, use the anchor tracks when helpful, and suggest 5-10 seed tracks to build from.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name: { type: Type.STRING },
            description: { type: Type.STRING },
            reasoning: { type: Type.STRING },
            seedTracks: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ["name", "description", "reasoning", "seedTracks"],
        },
      },
    });

    res.json(parseSuggestionResponse(response.text || ""));
  } catch (error) {
    console.error("Failed to generate AI suggestion:", error);
    res.status(500).json({
      error: getErrorMessage(error, "Failed to generate an AI playlist suggestion."),
    });
  }
});

app.get("/api/health", (_req, res) => {
  try {
    res.json({
      status: "ok",
      requestOrigin: resolveRequestOrigin(_req),
      resolvedSpotifyRedirect: resolveSpotifyRedirectUri(_req),
      spotifyRedirects: Array.from(getAllowedRedirectUris()),
    });
  } catch (error) {
    res.status(500).json({
      status: "error",
      error: getErrorMessage(error, "Failed to resolve Spotify redirect configuration."),
      spotifyRedirects: Array.from(getAllowedRedirectUris()),
    });
  }
});

app.get("*", (req, res, next) => {
  if (req.path.startsWith("/api/")) {
    return next();
  }

  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
