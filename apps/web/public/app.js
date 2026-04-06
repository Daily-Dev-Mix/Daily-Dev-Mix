const SESSION_TYPES = [
  {
    id: "study",
    label: "Study",
    emoji: "📚",
    blurb: "Capture the listening patterns that help you stay locked in through long reading and writing blocks.",
  },
  {
    id: "work",
    label: "Deep Work",
    emoji: "💻",
    blurb: "Record the tracks that support coding, research, planning, and other concentration-heavy work.",
  },
  {
    id: "gym",
    label: "Workout",
    emoji: "💪",
    blurb: "Separate your training sound from broad workout playlists and keep the energy profile specific.",
  },
  {
    id: "roadtrip",
    label: "Road Trip",
    emoji: "🚗",
    blurb: "Learn the pacing, lift, and familiarity that make a long drive feel right.",
  },
  {
    id: "chill",
    label: "Unwind",
    emoji: "🧘",
    blurb: "Preserve the softer, slower listening patterns you return to when you want to reset.",
  },
];

const STORAGE_KEYS = {
  tokens: "spotify_tokens",
  activeSession: "active_session",
  sessionHistory: "session_history",
  suggestion: "last_suggestion",
  suggestionContext: "last_suggestion_context",
  explorationLevel: "exploration_level",
  lastActivityTime: "last_activity_time",
  selectedSessionType: "selected_session_type",
  sessionSeeds: "session_seed_bank",
  anchorDrafts: "anchor_song_drafts",
  reminderSessionId: "session_reminder_session_id",
};

const elements = {
  connectBtn: document.getElementById("connect-btn"),
  logoutBtn: document.getElementById("logout-btn"),
  authStatus: document.getElementById("auth-status"),
  liveSummary: document.getElementById("live-summary"),
  sessionTypes: document.getElementById("session-types"),
  selectedSessionSummary: document.getElementById("selected-session-summary"),
  anchorSongsInput: document.getElementById("anchor-songs-input"),
  anchorSongsHint: document.getElementById("anchor-songs-hint"),
  startSessionBtn: document.getElementById("start-session-btn"),
  endSessionBtn: document.getElementById("end-session-btn"),
  explorationRange: document.getElementById("exploration-range"),
  explorationValue: document.getElementById("exploration-value"),
  sessionStatus: document.getElementById("session-status"),
  currentTrack: document.getElementById("current-track"),
  topTracksList: document.getElementById("top-tracks-list"),
  recentTracksList: document.getElementById("recent-tracks-list"),
  suggestionContent: document.getElementById("suggestion-content"),
  createPlaylistBtn: document.getElementById("create-playlist-btn"),
  profileSummary: document.getElementById("profile-summary"),
  historyList: document.getElementById("history-list"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
  toast: document.getElementById("toast"),
};

const requestCache = new Map();
const apiBaseUrlMeta = document.querySelector('meta[name="daily-dev-mix-api-base-url"]');
let toastTimer = null;
const REQUIRED_PLAYLIST_SCOPES = [
  "playlist-modify-public",
  "playlist-modify-private",
];

function isLoopbackHostname(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function normalizeApiBaseUrl(rawBaseUrl) {
  const trimmedBaseUrl = String(rawBaseUrl || "").trim();
  if (!trimmedBaseUrl) {
    return "";
  }

  return trimmedBaseUrl.replace(/\/+$/, "");
}

function resolveApiBaseUrl() {
  const explicitBaseUrl = normalizeApiBaseUrl(apiBaseUrlMeta?.content);
  if (explicitBaseUrl) {
    return explicitBaseUrl;
  }

  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  const origin = window.location.origin;

  if (protocol === "file:") {
    return "http://127.0.0.1:3000";
  }

  if (isLoopbackHostname(hostname)) {
    return "http://127.0.0.1:3000";
  }

  return origin;
}

const API_BASE_URL = resolveApiBaseUrl();
const TRUSTED_MESSAGE_ORIGINS = new Set([window.location.origin]);

try {
  TRUSTED_MESSAGE_ORIGINS.add(new URL(API_BASE_URL).origin);
} catch {
  // Ignore invalid API origins and fall back to the current page origin.
}

function buildApiUrl(path) {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  if (!path.startsWith("/")) {
    return `${API_BASE_URL}/${path}`;
  }

  return `${API_BASE_URL}${path}`;
}

function simplifyRequestErrorMessage(rawMessage, path) {
  const message = String(rawMessage || "Request failed.");
  const compactMessage = message.replace(/\s+/g, " ").trim();
  const isMissingRouteHtml = compactMessage.includes(`Cannot GET ${path}`);

  if (!isMissingRouteHtml) {
    return message;
  }

  return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. Start the Daily Dev Mix web server with npm start, or update the frontend API base URL.`;
}

function buildNetworkErrorMessage(path) {
  const currentProtocol = window.location.protocol;
  const currentHostname = window.location.hostname;
  const currentPort = window.location.port;
  const openedFromFile = currentProtocol === "file:";
  const usingLocalBackend = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(API_BASE_URL);
  const likelyProtocolMismatch =
    currentProtocol === "https:" && usingLocalBackend && isLoopbackHostname(currentHostname);

  if (openedFromFile) {
    return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. Start the Daily Dev Mix web server with npm start and open the app from http://127.0.0.1:3000 instead of opening index.html directly.`;
  }

  if (likelyProtocolMismatch) {
    return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. Your page is using HTTPS, but the local backend runs on HTTP. Open http://127.0.0.1:3000 to test locally.`;
  }

  if (isLoopbackHostname(currentHostname) && currentPort !== "3000") {
    return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. The app is open on ${window.location.origin}, so make sure the backend is running and that ${API_BASE_URL}/api/health loads in the browser.`;
  }

  if (usingLocalBackend) {
    return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. Make sure the Daily Dev Mix web server is running and that ${API_BASE_URL}/api/health returns JSON.`;
  }

  return `Daily Dev Mix could not reach ${path} on ${API_BASE_URL}. Check that the backend origin is correct and reachable.`;
}

function readStorage(key, fallback) {
  try {
    const rawValue = localStorage.getItem(key);
    return rawValue ? JSON.parse(rawValue) : fallback;
  } catch {
    return fallback;
  }
}

function normalizeSpotifyScopes(rawScopes) {
  if (Array.isArray(rawScopes)) {
    return rawScopes
      .map((value) => String(value || "").trim())
      .filter(Boolean);
  }

  if (typeof rawScopes === "string") {
    return rawScopes
      .split(/\s+/)
      .map((value) => value.trim())
      .filter(Boolean);
  }

  return [];
}

function normalizeTokens(rawTokens) {
  if (!rawTokens || typeof rawTokens !== "object") {
    return null;
  }

  const accessToken = typeof rawTokens.accessToken === "string" ? rawTokens.accessToken.trim() : "";
  const refreshToken = typeof rawTokens.refreshToken === "string" ? rawTokens.refreshToken.trim() : "";
  const expiresIn = Number(rawTokens.expiresIn);

  if (!accessToken && !refreshToken) {
    return null;
  }

  return {
    ...rawTokens,
    accessToken,
    refreshToken,
    expiresIn: Number.isFinite(expiresIn) ? expiresIn : undefined,
    scopes: normalizeSpotifyScopes(rawTokens.scopes),
  };
}

function getMissingSpotifyScopes(requiredScopes, grantedScopes) {
  const grantedScopeSet = new Set(normalizeSpotifyScopes(grantedScopes));
  return requiredScopes.filter((scope) => !grantedScopeSet.has(scope));
}

function writeStorage(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function removeStorage(key) {
  localStorage.removeItem(key);
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getMonogram(value, fallback = "DM") {
  const parts = String(value || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (!parts.length) {
    return fallback;
  }

  if (parts.length === 1) {
    return parts[0].slice(0, 2).toUpperCase();
  }

  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
}

function formatAccountTier(value) {
  const normalizedValue = String(value || "").trim();
  if (!normalizedValue) {
    return "Spotify account";
  }

  return normalizedValue.charAt(0).toUpperCase() + normalizedValue.slice(1);
}

function uniqueStrings(values) {
  const seen = new Set();
  const result = [];

  for (const rawValue of values) {
    const trimmedValue = String(rawValue || "").trim();
    if (!trimmedValue) {
      continue;
    }

    const key = trimmedValue.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    result.push(trimmedValue);
  }

  return result;
}

function splitAnchorSongs(rawValue) {
  return uniqueStrings(
    String(rawValue || "")
      .split(/\n|,/g)
      .map((value) => value.trim()),
  ).slice(0, 5);
}

function formatDateTime(timestamp) {
  return new Date(timestamp).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(startTime, endTime = Date.now()) {
  const totalMinutes = Math.max(1, Math.floor((endTime - startTime) / 60000));

  if (totalMinutes < 60) {
    return `${totalMinutes} min`;
  }

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes ? `${hours} hr ${minutes} min` : `${hours} hr`;
}

function normalizeRecordedTrack(track) {
  const artists = Array.isArray(track?.artists) && track.artists.length
    ? track.artists.map((artist) => ({ name: artist.name || "Unknown artist" }))
    : [{ name: "Unknown artist" }];

  const images = Array.isArray(track?.album?.images)
    ? track.album.images
        .map((image) => ({ url: image.url || "" }))
        .filter((image) => image.url)
    : [];

  return {
    id: track?.id || track?.uri || (window.crypto?.randomUUID?.() ?? `${Date.now()}`),
    name: track?.name || "Unknown track",
    uri: track?.uri || "",
    artists,
    album: {
      images,
    },
  };
}

function normalizeTrackSummary(track) {
  const normalizedTrack = normalizeRecordedTrack(track);
  return {
    id: normalizedTrack.id,
    name: normalizedTrack.name,
    artistLine: normalizedTrack.artists.map((artist) => artist.name).join(", "),
  };
}

function normalizeSession(session) {
  if (!session || typeof session !== "object") {
    return null;
  }

  return {
    id: session.id || (window.crypto?.randomUUID?.() ?? `${Date.now()}`),
    type: session.type || "study",
    startTime: Number(session.startTime || Date.now()),
    endTime: session.endTime ? Number(session.endTime) : undefined,
    tracks: Array.isArray(session.tracks) ? session.tracks.map((track) => normalizeRecordedTrack(track)) : [],
    anchorTracks: Array.isArray(session.anchorTracks) ? uniqueStrings(session.anchorTracks).slice(0, 5) : [],
  };
}

function getSessionType(typeId) {
  return SESSION_TYPES.find((type) => type.id === typeId);
}

const state = {
  tokens: normalizeTokens(readStorage(STORAGE_KEYS.tokens, null)),
  user: null,
  currentTrack: null,
  topTracks: [],
  recentTracks: [],
  activeSession: normalizeSession(readStorage(STORAGE_KEYS.activeSession, null)),
  sessionHistory: readStorage(STORAGE_KEYS.sessionHistory, [])
    .map((session) => normalizeSession(session))
    .filter(Boolean),
  suggestion: readStorage(STORAGE_KEYS.suggestion, null),
  suggestionContext: readStorage(STORAGE_KEYS.suggestionContext, null),
  explorationLevel: readStorage(STORAGE_KEYS.explorationLevel, 50),
  lastActivityTime: readStorage(STORAGE_KEYS.lastActivityTime, Date.now()),
  selectedSessionType: readStorage(STORAGE_KEYS.selectedSessionType, null),
  sessionSeeds: readStorage(STORAGE_KEYS.sessionSeeds, {}),
  anchorDrafts: readStorage(STORAGE_KEYS.anchorDrafts, {}),
  reminderSessionId: readStorage(STORAGE_KEYS.reminderSessionId, ""),
  isGenerating: false,
  pollTimer: null,
};

if (!state.selectedSessionType) {
  state.selectedSessionType = state.activeSession?.type || SESSION_TYPES[0].id;
}

function showToast(message, kind = "info") {
  if (!elements.toast) {
    return;
  }

  elements.toast.hidden = false;
  elements.toast.dataset.kind = kind;
  elements.toast.textContent = message;

  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3800);
}

function getRetryAfterMs(response, attempt) {
  const retryAfterHeader = response.headers.get("retry-after");
  const retryAfterSeconds = Number.parseInt(retryAfterHeader || "", 10);

  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds >= 0) {
    return retryAfterSeconds * 1000;
  }

  return Math.min(8000, 500 * 2 ** attempt);
}

function getCachedPayload(cacheKey) {
  if (!cacheKey) {
    return null;
  }

  const cachedValue = requestCache.get(cacheKey);
  if (!cachedValue) {
    return null;
  }

  if (cachedValue.expiresAt <= Date.now()) {
    requestCache.delete(cacheKey);
    return null;
  }

  return cachedValue.value;
}

function setCachedPayload(cacheKey, payload, ttlMs) {
  if (!cacheKey || !ttlMs) {
    return payload;
  }

  requestCache.set(cacheKey, {
    value: payload,
    expiresAt: Date.now() + ttlMs,
  });

  return payload;
}

async function refreshAccessToken() {
  if (!state.tokens?.refreshToken) {
    throw new Error("No Spotify refresh token is available.");
  }

  const response = await fetch(buildApiUrl("/api/auth/refresh"), {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ refreshToken: state.tokens.refreshToken }),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error || "Failed to refresh Spotify access token.");
  }

  state.tokens = {
    ...state.tokens,
    accessToken: payload.accessToken,
    refreshToken: payload.refreshToken || state.tokens.refreshToken,
    expiresIn: payload.expiresIn,
    scopes: normalizeSpotifyScopes(payload.scopes).length ? normalizeSpotifyScopes(payload.scopes) : state.tokens.scopes,
  };
  writeStorage(STORAGE_KEYS.tokens, state.tokens);
}

async function fetchJson(url, options = {}, config = {}) {
  const {
    requireAuth = true,
    retryOnAuthFailure = true,
    cacheKey = null,
    cacheTtlMs = 0,
    rateLimitRetries = 3,
  } = config;
  const method = options.method || "GET";

  if (method === "GET") {
    const cachedPayload = getCachedPayload(cacheKey);
    if (cachedPayload !== null) {
      return cachedPayload;
    }
  }

  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (requireAuth) {
    if (!state.tokens?.accessToken) {
      throw new Error("Connect Spotify before using this feature.");
    }

    headers.set("Authorization", `Bearer ${state.tokens.accessToken}`);
  }

  for (let attempt = 0; attempt <= rateLimitRetries; attempt += 1) {
    let response;

    try {
      response = await fetch(buildApiUrl(url), { ...options, headers, credentials: "include" });
    } catch (error) {
      const message = error instanceof Error ? error.message : "";
      const likelyNetworkError = !message || /failed to fetch|networkerror|load failed/i.test(message);

      if (likelyNetworkError) {
        throw new Error(buildNetworkErrorMessage(url));
      }

      throw error;
    }

    if (response.status === 401 && requireAuth && retryOnAuthFailure && state.tokens?.refreshToken) {
      await refreshAccessToken();
      return fetchJson(url, options, { ...config, retryOnAuthFailure: false });
    }

    if ((response.status === 429 || response.status === 503) && attempt < rateLimitRetries) {
      await delay(getRetryAfterMs(response, attempt));
      continue;
    }

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json().catch(() => null)
      : await response.text();

    if (!response.ok) {
      const errorMessage = payload?.error || payload || "Request failed.";
      throw new Error(simplifyRequestErrorMessage(errorMessage, url));
    }

    return method === "GET" ? setCachedPayload(cacheKey, payload, cacheTtlMs) : payload;
  }

  throw new Error("Request retry budget exhausted.");
}

function persistActiveSession() {
  if (state.activeSession) {
    writeStorage(STORAGE_KEYS.activeSession, state.activeSession);
  } else {
    removeStorage(STORAGE_KEYS.activeSession);
  }
}

function persistSuggestion() {
  if (state.suggestion) {
    writeStorage(STORAGE_KEYS.suggestion, state.suggestion);
  } else {
    removeStorage(STORAGE_KEYS.suggestion);
  }

  if (state.suggestionContext) {
    writeStorage(STORAGE_KEYS.suggestionContext, state.suggestionContext);
  } else {
    removeStorage(STORAGE_KEYS.suggestionContext);
  }
}

function persistDrafts() {
  writeStorage(STORAGE_KEYS.anchorDrafts, state.anchorDrafts);
}

function persistSeedBank() {
  writeStorage(STORAGE_KEYS.sessionSeeds, state.sessionSeeds);
}

function setSelectedSessionType(sessionTypeId) {
  state.selectedSessionType = sessionTypeId;
  writeStorage(STORAGE_KEYS.selectedSessionType, state.selectedSessionType);

  if (!state.anchorDrafts[sessionTypeId] && Array.isArray(state.sessionSeeds[sessionTypeId])) {
    state.anchorDrafts[sessionTypeId] = state.sessionSeeds[sessionTypeId].join("\n");
    persistDrafts();
  }

  renderSessionStudio();
  renderLiveSummary();
}

function getDraftTextForSession(sessionTypeId) {
  if (!sessionTypeId) {
    return "";
  }

  const storedDraft = state.anchorDrafts[sessionTypeId];
  if (typeof storedDraft === "string") {
    return storedDraft;
  }

  return Array.isArray(state.sessionSeeds[sessionTypeId]) ? state.sessionSeeds[sessionTypeId].join("\n") : "";
}

function renderLiveSummary() {
  const activeSessionType = getSessionType(state.activeSession?.type || state.selectedSessionType);
  const metrics = [
    {
      label: "Connection",
      value: state.user ? "Ready" : "Not connected",
    },
    {
      label: "Sessions",
      value: `${state.sessionHistory.length}`,
    },
    {
      label: "Session Type",
      value: activeSessionType?.label || "Not selected",
    },
  ];

  elements.liveSummary.innerHTML = metrics
    .map(
      (metric) => `
        <div class="metric-card">
          <strong>${escapeHtml(metric.value)}</strong>
          <span>${escapeHtml(metric.label)}</span>
        </div>
      `,
    )
    .join("");
}

function renderAuthStatus() {
  const isConnected = Boolean(state.user);
  elements.connectBtn.hidden = isConnected;
  elements.logoutBtn.hidden = !isConnected;
  elements.authStatus.textContent = isConnected
    ? `Connected as ${state.user.display_name || state.user.id}.`
    : "Connect your Spotify account to start capturing listening sessions.";
}

function renderProfile() {
  if (!state.user) {
    elements.profileSummary.className = "empty-state";
    elements.profileSummary.innerHTML = "Your Spotify account details will appear here after connection.";
    return;
  }

  const avatar = state.user.images?.[0]?.url || "";
  const displayName = state.user.display_name || state.user.id;
  const email = state.user.email || "Spotify account connected";
  const plan = formatAccountTier(state.user.product);
  const avatarMarkup = avatar
    ? `<img class="avatar" src="${escapeHtml(avatar)}" alt="Spotify avatar" referrerpolicy="no-referrer" />`
    : `<div class="avatar avatar-fallback" aria-hidden="true">${escapeHtml(getMonogram(displayName))}</div>`;

  elements.profileSummary.className = "profile-panel";
  elements.profileSummary.innerHTML = `
    ${avatarMarkup}
    <div>
      <strong>${escapeHtml(displayName)}</strong>
      <span>${escapeHtml(email)}</span>
      <div class="track-chip">${escapeHtml(plan)}</div>
    </div>
  `;
}

function renderSessionTiles() {
  const currentTypeId = state.activeSession?.type || state.selectedSessionType;

  elements.sessionTypes.innerHTML = SESSION_TYPES.map((type) => {
    const isSelected = currentTypeId === type.id;
    const isLive = state.activeSession?.type === type.id;

    return `
      <button
        class="session-tile${isSelected ? " is-selected" : ""}${isLive ? " is-live" : ""}"
        type="button"
        data-session-type="${escapeHtml(type.id)}"
      >
        <span class="emoji">${type.emoji}</span>
        <strong>${escapeHtml(type.label)}</strong>
        <span class="tile-copy">${escapeHtml(type.blurb)}</span>
      </button>
    `;
  }).join("");
}

function renderSelectedSessionSummary() {
  const activeOrSelectedType = getSessionType(state.activeSession?.type || state.selectedSessionType);

  if (!activeOrSelectedType) {
    elements.selectedSessionSummary.className = "empty-state";
    elements.selectedSessionSummary.innerHTML = "Select a session type to prepare your next listening capture.";
    return;
  }

  const savedSeeds = Array.isArray(state.sessionSeeds[activeOrSelectedType.id]) ? state.sessionSeeds[activeOrSelectedType.id] : [];
  const draftSeeds = splitAnchorSongs(getDraftTextForSession(activeOrSelectedType.id));
  const activeSeeds = state.activeSession?.anchorTracks || [];
  const previewSeeds = state.activeSession ? activeSeeds : (draftSeeds.length ? draftSeeds : savedSeeds);

  elements.selectedSessionSummary.className = "selected-summary";
  elements.selectedSessionSummary.innerHTML = `
    <div class="session-badge">${activeOrSelectedType.emoji} ${state.activeSession ? "Session active" : "Ready to start"}</div>
    <strong>${escapeHtml(activeOrSelectedType.label)}</strong>
    <span>${escapeHtml(activeOrSelectedType.blurb)}</span>
    <span>${previewSeeds.length} anchor track${previewSeeds.length === 1 ? "" : "s"} prepared for this session type.</span>
  `;
}

function renderAnchorEditor() {
  const activeOrSelectedTypeId = state.activeSession?.type || state.selectedSessionType;
  const savedSeeds = Array.isArray(state.sessionSeeds[activeOrSelectedTypeId]) ? state.sessionSeeds[activeOrSelectedTypeId] : [];
  const draftText = state.activeSession
    ? state.activeSession.anchorTracks.join("\n")
    : getDraftTextForSession(activeOrSelectedTypeId);
  const previewSeeds = state.activeSession ? state.activeSession.anchorTracks : splitAnchorSongs(draftText);

  if (elements.anchorSongsInput.value !== draftText) {
    elements.anchorSongsInput.value = draftText;
  }

  elements.anchorSongsInput.disabled = Boolean(state.activeSession);

  if (state.activeSession) {
    elements.anchorSongsHint.textContent = `Using ${previewSeeds.length} saved anchor track${previewSeeds.length === 1 ? "" : "s"} for the active session.`;
  } else if (!state.user) {
    elements.anchorSongsHint.textContent = "Connect Spotify, then add 3 to 5 anchor tracks for any new session type.";
  } else if (!savedSeeds.length && previewSeeds.length < 3) {
    elements.anchorSongsHint.textContent = "For a new session type, add at least 3 anchor tracks so the first recommendation has a reliable starting point.";
  } else {
    elements.anchorSongsHint.textContent = `${previewSeeds.length} anchor track${previewSeeds.length === 1 ? "" : "s"} ready. You can update them any time before starting a session.`;
  }

  elements.startSessionBtn.hidden = Boolean(state.activeSession);
  elements.endSessionBtn.hidden = !state.activeSession;
  elements.startSessionBtn.disabled = !state.user || !activeOrSelectedTypeId;
}

function renderSessionStatus() {
  if (!state.user) {
    elements.sessionStatus.className = "session-status empty-state";
    elements.sessionStatus.innerHTML = "Connect Spotify and choose a session type to begin.";
    return;
  }

  if (!state.activeSession) {
    const selectedType = getSessionType(state.selectedSessionType);
    elements.sessionStatus.className = "session-status empty-state";
    elements.sessionStatus.innerHTML = selectedType
      ? `Ready to capture a ${escapeHtml(selectedType.label.toLowerCase())} session. Start when you begin listening.`
      : "Choose a session type to start building a personalized playlist.";
    return;
  }

  const sessionType = getSessionType(state.activeSession.type);
  const anchorSongs = state.activeSession.anchorTracks || [];
  const needsReminderSoon = Date.now() - state.activeSession.startTime > 90 * 60 * 1000;

  elements.sessionStatus.className = "session-status session-summary";
  elements.sessionStatus.innerHTML = `
    <div class="session-badge">${sessionType?.emoji || "🎵"} Capture in progress</div>
    <strong>${escapeHtml(sessionType?.label || state.activeSession.type)}</strong>
    <span>Started ${escapeHtml(formatDateTime(state.activeSession.startTime))}</span>
    <span>${escapeHtml(formatDuration(state.activeSession.startTime))} elapsed</span>
    <span>${state.activeSession.tracks.length} unique tracks captured</span>
    <span>${anchorSongs.length} anchor track${anchorSongs.length === 1 ? "" : "s"} supporting this session type</span>
    <span>${needsReminderSoon ? "A reminder is queued if this session continues much longer." : "The session will auto-stop after 30 minutes without Spotify playback."}</span>
  `;
}

function renderCurrentTrack() {
  const currentTrack = state.currentTrack?.item;

  if (!currentTrack) {
    elements.currentTrack.className = "track-panel empty-state";
    elements.currentTrack.innerHTML = "Live playback details appear here while Spotify is active.";
    return;
  }

  const isCaptured = Boolean(
    state.activeSession?.tracks.some((track) => track.id === currentTrack.id || track.uri === currentTrack.uri),
  );

  const artworkUrl = currentTrack.album?.images?.[0]?.url || "";
  const artworkMarkup = artworkUrl
    ? `<img
        class="track-art"
        src="${escapeHtml(artworkUrl)}"
        alt="Album art"
        referrerpolicy="no-referrer"
      />`
    : `<div class="track-art track-art-fallback" aria-hidden="true">${escapeHtml(getMonogram(currentTrack.name, "MX"))}</div>`;

  elements.currentTrack.className = "track-panel";
  elements.currentTrack.innerHTML = `
    <div class="track-card">
      ${artworkMarkup}
      <div class="track-title">
        <strong>${escapeHtml(currentTrack.name)}</strong>
        <span>${escapeHtml(currentTrack.artists.map((artist) => artist.name).join(", "))}</span>
        <span>${escapeHtml(currentTrack.album?.name || "Spotify playback")}</span>
        <div class="track-chip">${state.currentTrack.is_playing ? "Playing now" : "Paused"}${isCaptured ? " • captured" : ""}</div>
      </div>
    </div>
  `;
}

function renderTasteList(container, tracks, emptyMessage) {
  if (!tracks.length) {
    container.className = "empty-state compact-state";
    container.innerHTML = emptyMessage;
    return;
  }

  container.className = "compact-list";
  container.innerHTML = tracks
    .map(
      (track) => `
        <span class="list-chip">${escapeHtml(track.name)} <span>${escapeHtml(track.artistLine)}</span></span>
      `,
    )
    .join("");
}

function renderTasteSnapshot() {
  renderTasteList(
    elements.topTracksList,
    state.topTracks,
    "Top tracks will appear here after Daily Dev Mix loads your long-term listening profile.",
  );
  renderTasteList(
    elements.recentTracksList,
    state.recentTracks,
    "Recent listening will appear here after Daily Dev Mix loads your short-term activity signal.",
  );
}

function renderSuggestion() {
  elements.createPlaylistBtn.hidden = !state.suggestion || state.isGenerating;

  if (state.isGenerating) {
    elements.suggestionContent.className = "suggestion-panel";
    elements.suggestionContent.innerHTML = `<div class="loading">Generating a playlist recommendation from this session...</div>`;
    return;
  }

  if (!state.suggestion) {
    elements.suggestionContent.className = "empty-state";
    elements.suggestionContent.innerHTML = "Finish a session to generate a playlist concept, rationale, and seed tracks.";
    return;
  }

  const context = state.suggestionContext || {};
  const contextLabel = context.sessionType ? getSessionType(context.sessionType)?.label || context.sessionType : "this activity";
  const anchorSummary = Array.isArray(context.anchorTracks) && context.anchorTracks.length
    ? `
        <p class="suggestion-label">Anchor tracks</p>
        <div class="seed-list">${context.anchorTracks.map((track) => `<span>${escapeHtml(track)}</span>`).join("")}</div>
      `
    : "";

  elements.suggestionContent.className = "suggestion-panel";
  elements.suggestionContent.innerHTML = `
    <h3>${escapeHtml(state.suggestion.name)}</h3>
    <p>${escapeHtml(state.suggestion.description)}</p>
    <p>${escapeHtml(state.suggestion.reasoning)}</p>
    <p>Built for ${escapeHtml(contextLabel)} from ${context.trackCount || 0} captured tracks with a ${context.explorationLevel ?? state.explorationLevel}% discovery setting.</p>
    ${anchorSummary}
    <p class="suggestion-label">Suggested seed tracks</p>
    <div class="seed-list">
      ${state.suggestion.seedTracks.map((track) => `<span>${escapeHtml(track)}</span>`).join("")}
    </div>
  `;
}

function renderHistory() {
  if (!state.sessionHistory.length) {
    elements.historyList.className = "history-list empty-state";
    elements.historyList.innerHTML = "No session history yet.";
    return;
  }

  elements.historyList.className = "history-list";
  elements.historyList.innerHTML = state.sessionHistory
    .map((session) => {
      const sessionType = getSessionType(session.type);
      const anchorCount = Array.isArray(session.anchorTracks) ? session.anchorTracks.length : 0;

      return `
        <article class="history-item">
          <strong>${escapeHtml(sessionType?.emoji || "🎵")} ${escapeHtml(sessionType?.label || session.type)}</strong>
          <div class="history-meta">
            <span>${escapeHtml(formatDateTime(session.startTime))}</span>
            <span>${escapeHtml(formatDuration(session.startTime, session.endTime))}</span>
          </div>
          <div class="history-chip">${session.tracks.length} track${session.tracks.length === 1 ? "" : "s"} captured</div>
          <div class="history-chip">${anchorCount} anchor song${anchorCount === 1 ? "" : "s"}</div>
        </article>
      `;
    })
    .join("");
}

function renderSessionStudio() {
  renderSessionTiles();
  renderSelectedSessionSummary();
  renderAnchorEditor();
  renderSessionStatus();
}

function renderApp() {
  elements.explorationRange.value = String(state.explorationLevel);
  elements.explorationValue.textContent = `${state.explorationLevel}%`;
  renderAuthStatus();
  renderLiveSummary();
  renderProfile();
  renderSessionStudio();
  renderCurrentTrack();
  renderTasteSnapshot();
  renderSuggestion();
  renderHistory();
}

function clearAuthState() {
  state.tokens = null;
  state.user = null;
  state.currentTrack = null;
  state.topTracks = [];
  state.recentTracks = [];
  requestCache.clear();
  removeStorage(STORAGE_KEYS.tokens);
}

function stopPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startPolling() {
  if (state.pollTimer) {
    return;
  }

  void pollSpotify();
  state.pollTimer = window.setInterval(() => {
    void pollSpotify();
  }, 5000);
}

function maybeSendReminder() {
  if (!state.activeSession) {
    return;
  }

  const unusuallyLongSession = Date.now() - state.activeSession.startTime >= 2 * 60 * 60 * 1000;
  if (!unusuallyLongSession || state.reminderSessionId === state.activeSession.id) {
    return;
  }

  state.reminderSessionId = state.activeSession.id;
  writeStorage(STORAGE_KEYS.reminderSessionId, state.reminderSessionId);

  const sessionType = getSessionType(state.activeSession.type);
  const message = `${sessionType?.label || "Your session"} has been running for a while. End it if you have switched activities.`;
  showToast(message, "info");

  if ("Notification" in window && Notification.permission === "granted") {
    new Notification("Daily Dev Mix reminder", { body: message });
  }
}

async function requestNotificationPermission() {
  if (!("Notification" in window) || Notification.permission !== "default") {
    return;
  }

  try {
    await Notification.requestPermission();
  } catch {
    // Ignore permission errors.
  }
}

async function loadTasteSnapshot() {
  if (!state.tokens?.accessToken) {
    state.topTracks = [];
    state.recentTracks = [];
    renderTasteSnapshot();
    return;
  }

  const [topTracksResult, recentTracksResult] = await Promise.allSettled([
    fetchJson("/api/spotify/top-tracks?limit=6", {}, {
      cacheKey: "spotify-top-tracks-6",
      cacheTtlMs: 10 * 60 * 1000,
    }),
    fetchJson("/api/spotify/recently-played?limit=6", {}, {
      cacheKey: "spotify-recently-played-6",
      cacheTtlMs: 60 * 1000,
    }),
  ]);

  state.topTracks = topTracksResult.status === "fulfilled"
    ? (topTracksResult.value.items || []).map((track) => normalizeTrackSummary(track))
    : [];

  state.recentTracks = recentTracksResult.status === "fulfilled"
    ? (recentTracksResult.value.items || []).map((item) => normalizeTrackSummary(item.track))
    : [];

  renderTasteSnapshot();
}

async function verifyConnection() {
  if (!state.tokens?.accessToken) {
    renderApp();
    return;
  }

  try {
    state.user = await fetchJson("/api/spotify/me", {}, {
      cacheKey: "spotify-profile",
      cacheTtlMs: 5 * 60 * 1000,
    });
    await loadTasteSnapshot();
    startPolling();
  } catch (error) {
    stopPolling();
    clearAuthState();
    showToast(error.message || "Your Spotify session expired. Please connect again.", "error");
  }

  renderApp();
}

async function connectSpotify() {
  try {
    const payload = await fetchJson("/api/auth/url", {}, { requireAuth: false });
    const popup = window.open(payload.url, "spotify_auth", "width=640,height=760");

    if (!popup) {
      throw new Error("Your browser blocked the Spotify sign-in window.");
    }
  } catch (error) {
    showToast(error.message || "Unable to start Spotify sign-in.", "error");
  }
}

function logout() {
  stopPolling();
  clearAuthState();
  renderApp();
}

function startSession() {
  const sessionTypeId = state.selectedSessionType;
  const savedSeeds = Array.isArray(state.sessionSeeds[sessionTypeId]) ? state.sessionSeeds[sessionTypeId] : [];
  const draftedSeeds = splitAnchorSongs(elements.anchorSongsInput.value);
  const anchorTracks = draftedSeeds.length ? draftedSeeds : savedSeeds;

  if (!state.user) {
    showToast("Connect Spotify before starting a session.", "error");
    return;
  }

  if (!sessionTypeId) {
    showToast("Choose a session type before starting.", "error");
    return;
  }

  if (!savedSeeds.length && anchorTracks.length < 3) {
    showToast("Add at least 3 anchor tracks for a new session type.", "error");
    return;
  }

  state.anchorDrafts[sessionTypeId] = anchorTracks.join("\n");
  state.sessionSeeds[sessionTypeId] = anchorTracks;
  persistDrafts();
  persistSeedBank();

  state.activeSession = {
    id: window.crypto?.randomUUID?.() || `${Date.now()}`,
    type: sessionTypeId,
    startTime: Date.now(),
    tracks: [],
    anchorTracks,
  };

  state.reminderSessionId = "";
  removeStorage(STORAGE_KEYS.reminderSessionId);
  state.suggestion = null;
  state.suggestionContext = null;
  state.lastActivityTime = Date.now();
  writeStorage(STORAGE_KEYS.lastActivityTime, state.lastActivityTime);
  persistActiveSession();
  persistSuggestion();
  void requestNotificationPermission();
  renderApp();
  showToast("Session started. Daily Dev Mix is now capturing this listening context.", "success");
}

async function generateSuggestion(session) {
  state.isGenerating = true;
  state.suggestion = null;
  state.suggestionContext = null;
  persistSuggestion();
  renderSuggestion();

  try {
    const suggestion = await fetchJson("/api/ai/suggestion", {
      method: "POST",
      body: JSON.stringify({
        sessionType: session.type,
        explorationLevel: state.explorationLevel,
        tracks: session.tracks,
        anchorTracks: session.anchorTracks || [],
      }),
    });

    state.suggestion = suggestion;
    state.suggestionContext = {
      sessionId: session.id,
      sessionType: session.type,
      trackCount: session.tracks.length,
      explorationLevel: state.explorationLevel,
      anchorTracks: session.anchorTracks || [],
      trackUris: session.tracks.map((track) => track.uri).filter(Boolean),
    };
    persistSuggestion();
    showToast("Your playlist recommendation is ready.", "success");
  } catch (error) {
    showToast(error.message || "We could not generate a playlist recommendation from that session.", "error");
  } finally {
    state.isGenerating = false;
    renderSuggestion();
  }
}

async function endSession(autoStopped = false) {
  if (!state.activeSession) {
    return;
  }

  const completedSession = {
    ...state.activeSession,
    endTime: Date.now(),
  };

  state.activeSession = null;
  state.reminderSessionId = "";
  removeStorage(STORAGE_KEYS.reminderSessionId);
  state.sessionHistory = [completedSession, ...state.sessionHistory].slice(0, 10);
  writeStorage(STORAGE_KEYS.sessionHistory, state.sessionHistory);
  persistActiveSession();
  renderApp();

  if (autoStopped) {
    showToast("This session ended automatically after 30 minutes without Spotify playback.", "info");
  }

  await generateSuggestion(completedSession);
}

async function createPlaylist() {
  if (!state.suggestion || !state.suggestionContext?.trackUris?.length) {
    showToast("This recommendation does not have enough captured Spotify tracks to create a playlist yet.", "error");
    return;
  }

  if (!state.tokens?.scopes?.length) {
    showToast("Reconnect Spotify once to refresh playlist permissions, then try saving again.", "error");
    return;
  }

  const missingPlaylistScopes = getMissingSpotifyScopes(REQUIRED_PLAYLIST_SCOPES, state.tokens.scopes);
  if (missingPlaylistScopes.length) {
    showToast(`Reconnect Spotify to grant ${missingPlaylistScopes.join(" and ")} access.`, "error");
    return;
  }

  try {
    await fetchJson("/api/spotify/create-playlist", {
      method: "POST",
      body: JSON.stringify({
        name: state.suggestion.name,
        description: state.suggestion.description,
        trackUris: state.suggestionContext.trackUris,
      }),
    });

    showToast("Playlist created in Spotify.", "success");
  } catch (error) {
    showToast(error.message || "We could not create the Spotify playlist.", "error");
  }
}

function clearHistory() {
  if (!state.sessionHistory.length) {
    return;
  }

  if (!window.confirm("Clear saved session history from this browser?")) {
    return;
  }

  state.sessionHistory = [];
  writeStorage(STORAGE_KEYS.sessionHistory, state.sessionHistory);
  renderHistory();
  renderLiveSummary();
}

async function pollSpotify() {
  if (!state.tokens?.accessToken) {
    stopPolling();
    return;
  }

  try {
    const playback = await fetchJson("/api/spotify/currently-playing", {}, {
      cacheKey: null,
      cacheTtlMs: 0,
      rateLimitRetries: 2,
    });
    state.currentTrack = playback;

    if (playback?.is_playing) {
      state.lastActivityTime = Date.now();
      writeStorage(STORAGE_KEYS.lastActivityTime, state.lastActivityTime);
    }

    if (state.activeSession && playback?.item && playback.is_playing) {
      const alreadyRecorded = state.activeSession.tracks.some(
        (track) => track.id === playback.item.id || track.uri === playback.item.uri,
      );

      if (!alreadyRecorded) {
        state.activeSession.tracks.push(normalizeRecordedTrack(playback.item));
        persistActiveSession();
      }
    }

    if (state.activeSession) {
      maybeSendReminder();

      if (Date.now() - Number(state.lastActivityTime || 0) > 30 * 60 * 1000) {
        await endSession(true);
        return;
      }
    }
  } catch (error) {
    showToast(error.message || "Unable to refresh live Spotify playback.", "error");
  }

  renderCurrentTrack();
  renderSessionStatus();
  renderHistory();
}

window.addEventListener("message", async (event) => {
  if (!TRUSTED_MESSAGE_ORIGINS.has(event.origin)) {
    return;
  }

  if (event.data?.type === "SPOTIFY_AUTH_SUCCESS") {
    state.tokens = normalizeTokens(event.data.tokens);
    writeStorage(STORAGE_KEYS.tokens, state.tokens);
    showToast("Spotify connected.", "success");
    await verifyConnection();
  }

  if (event.data?.type === "SPOTIFY_AUTH_ERROR") {
    showToast(event.data.message || "Spotify authentication failed.", "error");
  }
});

elements.connectBtn.addEventListener("click", connectSpotify);
elements.logoutBtn.addEventListener("click", logout);
elements.startSessionBtn.addEventListener("click", startSession);
elements.endSessionBtn.addEventListener("click", () => {
  void endSession(false);
});
elements.createPlaylistBtn.addEventListener("click", () => {
  void createPlaylist();
});
elements.clearHistoryBtn.addEventListener("click", clearHistory);

elements.explorationRange.addEventListener("input", (event) => {
  state.explorationLevel = Number(event.target.value);
  writeStorage(STORAGE_KEYS.explorationLevel, state.explorationLevel);
  elements.explorationValue.textContent = `${state.explorationLevel}%`;
  renderLiveSummary();
});

elements.anchorSongsInput.addEventListener("input", (event) => {
  const sessionTypeId = state.activeSession?.type || state.selectedSessionType;
  if (!sessionTypeId || state.activeSession) {
    return;
  }

  state.anchorDrafts[sessionTypeId] = event.target.value;
  persistDrafts();
  renderAnchorEditor();
  renderSelectedSessionSummary();
});

elements.sessionTypes.addEventListener("click", (event) => {
  const button = event.target.closest("[data-session-type]");
  if (!button) {
    return;
  }

  setSelectedSessionType(button.dataset.sessionType);
});

renderApp();
void verifyConnection();
