require('dotenv').config();
console.log('REDIRECT URI:', process.env.SPOTIFY_REDIRECT_URI);

// Import Dependencies -->
const crypto = require('crypto');
const express = require('express'); // To build an application server or API
const app = express();
const handlebars = require('express-handlebars'); //to enable express to work with handlebars
const Handlebars = require('handlebars'); // to include the templating engine responsible for compiling templates
const path = require('path');
const bodyParser = require('body-parser');
const session = require('express-session'); // To set the session object. To store or access session data, use the `req.session`, which is (generally) serialized as JSON by the store.
const axios = require('axios'); // To make HTTP requests from our server. We'll learn more about it in Part C.

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '0.0.0.0';
const SESSION_SECRET = process.env.SESSION_SECRET || 'local-dev-session-secret';
const REQUIRED_SPOTIFY_SCOPES = [
  'user-read-private',
  'user-read-currently-playing',
  'user-read-playback-state',
  'user-read-recently-played',
  'playlist-read-private',
  'playlist-modify-public',
  'playlist-modify-private',
];
const SPOTIFY_SCOPE = REQUIRED_SPOTIFY_SCOPES.join(' ');
const SPOTIFY_API_BASE_URL = 'https://api.spotify.com/v1';
const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token';
const PLAYLIST_MARKER = 'Created by Daily Dev Mix.';
const MAX_RECENT_PLAY_PAGES = 5;

if (!process.env.SESSION_SECRET) {
  console.warn('SESSION_SECRET is not set. Using a local development fallback secret.');
}

//Connect to DB -->
// create `ExpressHandlebars` instance and configure the layouts and partials dir.
const hbs = handlebars.create({
  extname: 'hbs',
  layoutsDir: __dirname + '/views/layouts',
  partialsDir: __dirname + '/views/partials',
});

// App Settings -->
// Register `hbs` as our view engine using its bound `engine()` function.
app.engine('hbs', hbs.engine);
app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'views'));
app.use(bodyParser.json()); // specify the usage of JSON for parsing request body.
// initialize session variables
app.use(
  session({
    secret: SESSION_SECRET,
    saveUninitialized: false,
    resave: false,
  })
);
app.use(
  bodyParser.urlencoded({
    extended: true,
  })
);

//helper for handlebars to see if strings or vars are equal
Handlebars.registerHelper('eq', (a, b) => a === b);

function isSpotifyConfigured() {
  return Boolean(
    process.env.SPOTIFY_CLIENT_ID &&
      process.env.SPOTIFY_CLIENT_SECRET &&
      process.env.SPOTIFY_REDIRECT_URI
  );
}

function formatTrackDuration(ms) {
  const totalSeconds = Math.max(0, Math.floor((ms || 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function formatSessionDuration(ms) {
  const totalSeconds = Math.max(0, Math.floor((ms || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m ${seconds}s`;
}

function buildSpotifyDesktopUrl(type, id) {
  if (!id) {
    return '';
  }

  return `spotify:${type}:${id}`;
}

function getActiveSessionDuration(sessionData) {
  const endTime = sessionData.endedAt || Date.now();
  return Math.max(0, endTime - sessionData.startedAt);
}

function formatSpotifyError(error) {
  const status = error.response?.status;
  const description =
    error.response?.data?.error_description ||
    error.response?.data?.error?.message ||
    error.response?.data?.error ||
    error.message;

  return `Spotify request failed${status ? ` (${status})` : ''}: ${description}`;
}

function getSpotifyErrorDescription(error) {
  return (
    error.response?.data?.error?.message ||
    error.response?.data?.error_description ||
    error.response?.data?.error ||
    error.message ||
    ''
  );
}

function isGenericSpotifyErrorDescription(description) {
  if (!description) {
    return true;
  }

  return (
    description === 'Request failed with status code 403' ||
    description === 'Request failed with status code 401'
  );
}

function getMissingSpotifyScopes(req) {
  const grantedScopes = req.session.spotifyGrantedScopes || [];
  return REQUIRED_SPOTIFY_SCOPES.filter(scope => !grantedScopes.includes(scope));
}

function getSpotifyRouteError(req, error, fallbackMessage) {
  const status = error.response?.status;
  const spotifyDescription = getSpotifyErrorDescription(error);
  const missingScopes = getMissingSpotifyScopes(req);

  if (status === 401) {
    return {
      status,
      message: 'Your Spotify login expired. Please sign in with Spotify again.',
    };
  }

  if (status === 403) {
    let message = !isGenericSpotifyErrorDescription(spotifyDescription)
      ? `Spotify rejected the request: ${spotifyDescription}.`
      : 'Spotify rejected the request.';

    if (missingScopes.length) {
      message += ` Missing scopes: ${missingScopes.join(', ')}.`;
    }

    if (isGenericSpotifyErrorDescription(spotifyDescription) && !missingScopes.length) {
      message +=
        ' If your app is still in Spotify development mode, make sure this Spotify account is in the app allowlist and that the app owner has Spotify Premium.';
    }

    message += ' Try signing in again and approving the requested Spotify permissions.';

    return {
      status,
      message,
    };
  }

  if (status === 429) {
    return {
      status,
      message: 'Spotify is rate limiting requests right now. Please wait a moment and try again.',
    };
  }

  return {
    status: 502,
    message: fallbackMessage,
  };
}

function sessionTrackMatches(existingTrack, nextTrack) {
  if (existingTrack.id !== nextTrack.id) {
    return false;
  }

  const toleranceMs = Math.min(
    15000,
    Math.max(existingTrack.durationMs || 0, nextTrack.durationMs || 0, 5000)
  );

  return (
    nextTrack.startedAt <= existingTrack.completedAt + toleranceMs &&
    nextTrack.completedAt >= existingTrack.startedAt - toleranceMs
  );
}

function findMatchingSessionTrack(sessionData, nextTrack) {
  for (let index = sessionData.tracks.length - 1; index >= 0; index -= 1) {
    const existingTrack = sessionData.tracks[index];

    if (sessionTrackMatches(existingTrack, nextTrack)) {
      return existingTrack;
    }
  }

  return null;
}

function mergeSessionTrack(existingTrack, nextTrack) {
  existingTrack.startedAt = Math.min(existingTrack.startedAt, nextTrack.startedAt);
  existingTrack.completedAt = Math.max(existingTrack.completedAt, nextTrack.completedAt);
  existingTrack.durationMs = Math.max(existingTrack.durationMs || 0, nextTrack.durationMs || 0);
  existingTrack.externalUrl = existingTrack.externalUrl || nextTrack.externalUrl || '';
  existingTrack.desktopUrl = existingTrack.desktopUrl || nextTrack.desktopUrl || '';
  existingTrack.albumArt = existingTrack.albumArt || nextTrack.albumArt || '';
  existingTrack.album = existingTrack.album || nextTrack.album || 'Unknown Album';
  existingTrack.artist = existingTrack.artist || nextTrack.artist || '';
  existingTrack.artists = existingTrack.artists?.length
    ? existingTrack.artists
    : nextTrack.artists || [];
}

function cloneSessionTrack(track) {
  return {
    ...track,
    artists: Array.isArray(track.artists) ? track.artists.slice() : [],
  };
}

function normalizeSessionTracks(sessionData) {
  const normalizedTracks = [];

  sessionData.tracks
    .slice()
    .sort((a, b) => a.startedAt - b.startedAt)
    .forEach(track => {
      const nextTrack = cloneSessionTrack(track);
      const matchingTrack = findMatchingSessionTrack({ tracks: normalizedTracks }, nextTrack);

      if (matchingTrack) {
        mergeSessionTrack(matchingTrack, nextTrack);
        return;
      }

      normalizedTracks.push(nextTrack);
    });

  sessionData.tracks = normalizedTracks;
}

function buildTrackRecord(track, startedAt, completedAt) {
  return {
    id: track.id,
    uri: track.uri,
    name: track.name,
    artist: (track.artists || []).map(artist => artist.name).join(', '),
    artists: (track.artists || []).map(artist => artist.name),
    album: track.album?.name || 'Unknown Album',
    albumArt: track.album?.images?.[0]?.url || '',
    durationMs: track.duration_ms || 0,
    externalUrl: track.external_urls?.spotify || '',
    desktopUrl: buildSpotifyDesktopUrl('track', track.id),
    startedAt,
    completedAt,
  };
}

function appendTrackToSession(sessionData, nextTrack) {
  if (!nextTrack?.id) {
    return;
  }

  const matchingTrack = findMatchingSessionTrack(sessionData, nextTrack);

  if (matchingTrack) {
    mergeSessionTrack(matchingTrack, nextTrack);
    return;
  }

  sessionData.tracks.push(nextTrack);
  sessionData.tracks.sort((a, b) => a.startedAt - b.startedAt);
}

function buildCurrentTrackPayload(trackRecord, progressMs, isPlaying) {
  return {
    ...trackRecord,
    progressMs,
    progressLabel: formatTrackDuration(progressMs),
    durationLabel: formatTrackDuration(trackRecord.durationMs),
    isPlaying,
  };
}

function serializePlaylist(playlist) {
  return {
    id: playlist.id,
    name: playlist.name,
    description: playlist.description || '',
    imageUrl: playlist.imageUrl || '',
    trackCount: playlist.trackCount || 0,
    externalUrl: playlist.externalUrl || '',
    desktopUrl: playlist.desktopUrl || '',
    isPublic: Boolean(playlist.isPublic),
  };
}

function serializeSession(sessionData) {
  normalizeSessionTracks(sessionData);

  const durationMs = getActiveSessionDuration(sessionData);
  const uniqueTrackCount = new Set(sessionData.tracks.map(track => track.id)).size;
  const uniqueArtistCount = new Set(
    sessionData.tracks.flatMap(track => track.artists || [])
  ).size;

  let playbackStatus = 'Idle';
  if (sessionData.currentTrack?.isPlaying) {
    playbackStatus = 'Live';
  } else if (sessionData.currentTrack) {
    playbackStatus = 'Paused';
  }

  return {
    id: sessionData.id,
    label: sessionData.label,
    emoji: sessionData.emoji,
    startedAt: sessionData.startedAt,
    endedAt: sessionData.endedAt || null,
    durationMs,
    durationLabel: formatSessionDuration(durationMs),
    trackCount: sessionData.tracks.length,
    uniqueTrackCount,
    uniqueArtistCount,
    playbackStatus,
    currentTrack: sessionData.currentTrack || null,
    tracks: sessionData.tracks
      .slice()
      .sort((a, b) => b.startedAt - a.startedAt)
      .map(track => ({
        ...track,
        durationLabel: formatTrackDuration(track.durationMs),
      })),
    playlist: sessionData.playlist ? serializePlaylist(sessionData.playlist) : null,
  };
}

function serializeHistorySession(sessionData) {
  const serializedSession = serializeSession(sessionData);
  return {
    id: serializedSession.id,
    label: serializedSession.label,
    emoji: serializedSession.emoji,
    startedAt: serializedSession.startedAt,
    endedAt: serializedSession.endedAt,
    durationMs: serializedSession.durationMs,
    durationLabel: serializedSession.durationLabel,
    trackCount: serializedSession.trackCount,
    uniqueArtistCount: serializedSession.uniqueArtistCount,
    playlist: serializedSession.playlist,
  };
}

function serializeSpotifyProfile(profile) {
  if (!profile) {
    return null;
  }

  return {
    id: profile.id || null,
    displayName: profile.display_name || null,
    product: profile.product || null,
    country: profile.country || null,
  };
}

function getPlaylistName(sessionData) {
  const dateStamp = new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
  }).format(new Date(sessionData.startedAt));

  return `Daily Dev Mix - ${sessionData.label} - ${dateStamp}`;
}

function getPlaylistDescription(sessionData, trackCount) {
  return `Generated from your ${sessionData.label} listening session with ${trackCount} tracked songs. ${PLAYLIST_MARKER}`;
}

function createListeningSession(label, emoji) {
  return {
    id: crypto.randomUUID(),
    label,
    emoji,
    startedAt: Date.now(),
    endedAt: null,
    tracks: [],
    currentTrack: null,
    playlist: null,
  };
}

function saveRequestSession(req) {
  return new Promise((resolve, reject) => {
    req.session.save(error => {
      if (error) {
        reject(error);
        return;
      }

      resolve();
    });
  });
}

function getSpotifyBasicAuthHeader() {
  return `Basic ${Buffer.from(
    `${process.env.SPOTIFY_CLIENT_ID}:${process.env.SPOTIFY_CLIENT_SECRET}`
  ).toString('base64')}`;
}

async function exchangeSpotifyToken(params) {
  return axios.post(SPOTIFY_TOKEN_URL, new URLSearchParams(params), {
    headers: {
      Authorization: getSpotifyBasicAuthHeader(),
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
}

async function refreshSpotifyAccessToken(req) {
  if (!req.session.spotifyRefreshToken) {
    throw new Error('Missing Spotify refresh token.');
  }

  const response = await exchangeSpotifyToken({
    grant_type: 'refresh_token',
    refresh_token: req.session.spotifyRefreshToken,
  });

  req.session.spotifyToken = response.data.access_token;
  req.session.spotifyTokenExpiresAt =
    Date.now() + Math.max((response.data.expires_in || 3600) - 60, 60) * 1000;

  if (response.data.refresh_token) {
    req.session.spotifyRefreshToken = response.data.refresh_token;
  }

  await saveRequestSession(req);
  return req.session.spotifyToken;
}

async function getSpotifyAccessToken(req) {
  if (!req.session.spotifyToken) {
    throw new Error('Missing Spotify access token.');
  }

  if (
    req.session.spotifyTokenExpiresAt &&
    Date.now() >= req.session.spotifyTokenExpiresAt
  ) {
    return refreshSpotifyAccessToken(req);
  }

  return req.session.spotifyToken;
}

async function spotifyApiRequest(req, config, canRetry = true) {
  const accessToken = await getSpotifyAccessToken(req);

  try {
    return await axios({
      baseURL: SPOTIFY_API_BASE_URL,
      ...config,
      headers: {
        Authorization: `Bearer ${accessToken}`,
        ...(config.headers || {}),
      },
    });
  } catch (error) {
    const shouldRetry =
      canRetry &&
      error.response?.status === 401 &&
      req.session.spotifyRefreshToken;

    if (!shouldRetry) {
      throw error;
    }

    await refreshSpotifyAccessToken(req);
    return spotifyApiRequest(req, config, false);
  }
}

async function getCurrentlyPlayingTrack(req) {
  const response = await spotifyApiRequest(req, {
    method: 'get',
    url: '/me/player/currently-playing',
  });

  if (response.status === 204 || !response.data?.item) {
    return null;
  }

  if (response.data.currently_playing_type !== 'track') {
    return null;
  }

  const progressMs = response.data.progress_ms || 0;
  const durationMs = response.data.item.duration_ms || 0;
  const startedAt = Math.max(0, (response.data.timestamp || Date.now()) - progressMs);
  const completedAt = startedAt + durationMs;

  return {
    isPlaying: Boolean(response.data.is_playing),
    progressMs,
    track: buildTrackRecord(response.data.item, startedAt, completedAt),
  };
}

async function getCurrentSpotifyProfile(req) {
  const response = await spotifyApiRequest(req, {
    method: 'get',
    url: '/me',
  });

  return response.data;
}

async function getRecentlyPlayedTracks(req, afterTimestamp) {
  let cursorAfter = afterTimestamp;
  const trackedItems = [];

  for (let page = 0; page < MAX_RECENT_PLAY_PAGES; page += 1) {
    const response = await spotifyApiRequest(req, {
      method: 'get',
      url: '/me/player/recently-played',
      params: {
        limit: 50,
        after: cursorAfter,
      },
    });

    const items = response.data?.items || [];

    if (!items.length) {
      break;
    }

    items.forEach(item => {
      if (!item.track?.id || !item.played_at) {
        return;
      }

      const completedAt = Date.parse(item.played_at);
      const durationMs = item.track.duration_ms || 0;
      const startedAt = Math.max(0, completedAt - durationMs);

      trackedItems.push(buildTrackRecord(item.track, startedAt, completedAt));
    });

    const nextCursor = Number(response.data?.cursors?.after);
    if (!nextCursor || nextCursor <= cursorAfter || items.length < 50) {
      break;
    }

    cursorAfter = nextCursor;
  }

  return trackedItems.sort((a, b) => a.startedAt - b.startedAt);
}

async function syncActiveSession(req) {
  const sessionData = req.session.activeListeningSession;

  if (!sessionData) {
    return null;
  }

  normalizeSessionTracks(sessionData);

  const currentPlayback = await getCurrentlyPlayingTrack(req);

  if (!currentPlayback?.track) {
    sessionData.currentTrack = null;
    return sessionData;
  }

  if (currentPlayback.isPlaying) {
    appendTrackToSession(sessionData, currentPlayback.track);
  }

  sessionData.currentTrack = buildCurrentTrackPayload(
    currentPlayback.track,
    currentPlayback.progressMs,
    currentPlayback.isPlaying
  );

  return sessionData;
}

async function enrichSessionWithRecentTracks(req, sessionData) {
  const recentTracks = await getRecentlyPlayedTracks(req, sessionData.startedAt - 1000);
  recentTracks.forEach(track => appendTrackToSession(sessionData, track));
}

async function createSpotifyPlaylist(req, sessionData) {
  const uniqueTracks = Array.from(
    new Map(
      sessionData.tracks
        .filter(track => track.uri)
        .map(track => [track.uri, track])
    ).values()
  );

  if (!uniqueTracks.length) {
    throw new Error('No songs were captured for this session yet.');
  }

  const createResponse = await spotifyApiRequest(req, {
    method: 'post',
    url: '/me/playlists',
    data: {
      name: getPlaylistName(sessionData),
      description: getPlaylistDescription(sessionData, uniqueTracks.length),
      public: false,
    },
  });

  const playlistId = createResponse.data.id;
  const uris = uniqueTracks.map(track => track.uri);

  for (let index = 0; index < uris.length; index += 100) {
    await spotifyApiRequest(req, {
      method: 'post',
      url: `/playlists/${playlistId}/items`,
      data: {
        uris: uris.slice(index, index + 100),
      },
    });
  }

  return {
    id: createResponse.data.id,
    name: createResponse.data.name,
    description: createResponse.data.description,
    imageUrl: sessionData.tracks[0]?.albumArt || '',
    trackCount: uniqueTracks.length,
    externalUrl: createResponse.data.external_urls?.spotify || '',
    desktopUrl: buildSpotifyDesktopUrl('playlist', createResponse.data.id),
    isPublic: Boolean(createResponse.data.public),
  };
}

async function getGeneratedPlaylists(req) {
  const response = await spotifyApiRequest(req, {
    method: 'get',
    url: '/me/playlists',
    params: {
      limit: 50,
    },
  });

  const localPlaylistsById = new Map(
    (req.session.generatedPlaylists || []).map(playlist => [playlist.id, playlist])
  );

  return (response.data?.items || [])
    .filter(playlist => (playlist.description || '').includes(PLAYLIST_MARKER))
    .map(playlist => {
      const localPlaylist = localPlaylistsById.get(playlist.id);
      return {
        id: playlist.id,
        name: playlist.name,
        description: playlist.description || '',
        imageUrl: playlist.images?.[0]?.url || localPlaylist?.imageUrl || '',
        trackCount: playlist.tracks?.total || localPlaylist?.trackCount || 0,
        externalUrl: playlist.external_urls?.spotify || '',
        desktopUrl:
          localPlaylist?.desktopUrl || buildSpotifyDesktopUrl('playlist', playlist.id),
        isPublic: Boolean(playlist.public),
      };
    });
}

// Authentication Middleware: require user to be logged in
const authPage = (req, res, next) => {
  if (!req.session.user) {
    return res.redirect('/login');
  }

  next();
};

const authApi = (req, res, next) => {
  if (!req.session.user || !req.session.spotifyToken) {
    return res.status(401).json({
      status: 'error',
      message: 'Please log in with Spotify first.',
    });
  }

  next();
};

//START OF API ROUTES
app.get('/', (req, res) => {
  if (req.session.user) {
    return res.redirect('/home');
  }
  res.render('pages/landing-page');
});

app.get('/login', (req, res) => {
  res.render('pages/login');
});
//Send Spotify data
app.get('/auth/spotify', async (req, res) => {
  if (!isSpotifyConfigured()) {
    return res.status(500).render('pages/login', {
      message: 'Spotify credentials are missing from the local environment.',
      error: true,
    });
  }

  const state = crypto.randomBytes(16).toString('hex');
  req.session.spotifyAuthState = state;
  await saveRequestSession(req);

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: process.env.SPOTIFY_CLIENT_ID,
    scope: SPOTIFY_SCOPE,
    redirect_uri: process.env.SPOTIFY_REDIRECT_URI,
    state,
    show_dialog: 'true',
  });

  res.redirect(`https://accounts.spotify.com/authorize?${params.toString()}`);
});

async function handleSpotifyCallback(req, res) {
  const { code, state, error } = req.query;

  if (error) {
    console.log('Spotify auth error:', error);
    return res.redirect('/login');
  }

  if (!code || state !== req.session.spotifyAuthState) {
    return res.redirect('/login');
  }

  try {
    const response = await exchangeSpotifyToken({
      grant_type: 'authorization_code',
      code,
      redirect_uri: process.env.SPOTIFY_REDIRECT_URI,
    });

    req.session.spotifyToken = response.data.access_token;
    req.session.spotifyRefreshToken =
      response.data.refresh_token || req.session.spotifyRefreshToken || null;
    req.session.spotifyTokenExpiresAt =
      Date.now() + Math.max((response.data.expires_in || 3600) - 60, 60) * 1000;
    req.session.spotifyGrantedScopes = String(response.data.scope || SPOTIFY_SCOPE)
      .split(' ')
      .map(scope => scope.trim())
      .filter(Boolean);
    req.session.user = { authenticated: true };
    req.session.generatedPlaylists = req.session.generatedPlaylists || [];
    req.session.sessionHistory = req.session.sessionHistory || [];
    delete req.session.spotifyAuthState;

    try {
      const spotifyProfile = await getCurrentSpotifyProfile(req);
      req.session.spotifyProfile = serializeSpotifyProfile(spotifyProfile);
    } catch (profileError) {
      console.log('Spotify profile load error:', formatSpotifyError(profileError));
    }

    await saveRequestSession(req);
    res.redirect('/home');
  } catch (requestError) {
    console.log('Spotify auth error:', formatSpotifyError(requestError));
    res.redirect('/login');
  }
}

//Recieve returned data from spotify
app.get('/auth/spotify/callback', handleSpotifyCallback);
app.get('/auth/callback', handleSpotifyCallback);

app.get('/home', authPage, (req, res) => {
  res.render('pages/home', { activePage: 'home' });
});

//Welcome route for lab 10
app.get('/welcome', (req, res) => {
  res.json({ status: 'success', message: 'Welcome!' });
});

app.get('/playlists', authPage, (req, res) => {
  res.render('pages/playlists', { activePage: 'playlists' });
});

app.get('/history', authPage, (req, res) => {
  res.render('pages/history', { activePage: 'history' });
});

app.get('/active-session', authPage, (req, res) => {
  res.render('pages/active-session');
});

app.post('/api/session/start', authApi, async (req, res) => {
  const label = String(req.body.label || '').trim();
  const emoji = String(req.body.emoji || '🎵').trim() || '🎵';
  const forceRestart = Boolean(req.body.forceRestart);

  if (!label) {
    return res.status(400).json({
      status: 'error',
      message: 'Choose a session vibe before starting.',
    });
  }

  let activeSession = req.session.activeListeningSession;
  let created = false;

  if (!activeSession || forceRestart) {
    activeSession = createListeningSession(label, emoji);
    req.session.activeListeningSession = activeSession;
    created = true;
  }

  let warning = null;

  try {
    await syncActiveSession(req);
  } catch (syncError) {
    console.log('Spotify session start sync error:', formatSpotifyError(syncError));
    const handledError = getSpotifyRouteError(
      req,
      syncError,
      'The session started, but Spotify playback could not be checked yet.'
    );

    if (handledError.status === 401) {
      return res.status(handledError.status).json({
        status: 'error',
        message: handledError.message,
      });
    }

    warning = handledError.message;
  }

  await saveRequestSession(req);

  res.status(created ? 201 : 200).json({
    status: 'success',
    resumed: !created,
    warning,
    session: serializeSession(req.session.activeListeningSession),
  });
});

app.get('/api/session/active', authApi, async (req, res) => {
  const activeSession = req.session.activeListeningSession;

  if (!activeSession) {
    return res.status(404).json({
      status: 'error',
      message: 'No active listening session was found.',
    });
  }

  let warning = null;

  try {
    await syncActiveSession(req);
    await saveRequestSession(req);
  } catch (syncError) {
    console.log('Spotify active session sync error:', formatSpotifyError(syncError));
    const handledError = getSpotifyRouteError(
      req,
      syncError,
      'Spotify playback could not be refreshed just now.'
    );

    if (handledError.status === 401) {
      return res.status(handledError.status).json({
        status: 'error',
        message: handledError.message,
      });
    }

    warning = handledError.message;
  }

  res.json({
    status: 'success',
    warning,
    session: serializeSession(req.session.activeListeningSession),
  });
});

app.post('/api/session/end', authApi, async (req, res) => {
  const activeSession = req.session.activeListeningSession;

  if (!activeSession) {
    return res.status(404).json({
      status: 'error',
      message: 'There is no active session to finish.',
    });
  }

  try {
    await syncActiveSession(req);
    await enrichSessionWithRecentTracks(req, activeSession);

    if (!activeSession.tracks.length) {
      return res.status(400).json({
        status: 'error',
        message: 'No tracks have been captured yet. Start playback in Spotify before creating a playlist.',
      });
    }

    activeSession.endedAt = Date.now();
    activeSession.playlist = await createSpotifyPlaylist(req, activeSession);

    const serializedPlaylist = serializePlaylist(activeSession.playlist);
    const serializedHistorySession = serializeHistorySession(activeSession);

    req.session.generatedPlaylists = [
      serializedPlaylist,
      ...(req.session.generatedPlaylists || []).filter(
        playlist => playlist.id !== serializedPlaylist.id
      ),
    ];

    req.session.sessionHistory = [
      serializedHistorySession,
      ...(req.session.sessionHistory || []).filter(
        sessionEntry => sessionEntry.id !== serializedHistorySession.id
      ),
    ].slice(0, 25);

    delete req.session.activeListeningSession;
    await saveRequestSession(req);

    res.json({
      status: 'success',
      playlist: serializedPlaylist,
      session: serializedHistorySession,
    });
  } catch (playlistError) {
    console.log('Spotify session end error:', formatSpotifyError(playlistError));
    const handledError = getSpotifyRouteError(
      req,
      playlistError,
      'The session could not be turned into a Spotify playlist.'
    );

    res.status(handledError.status).json({
      status: 'error',
      message: handledError.message,
    });
  }
});

app.get('/api/playlists', authApi, async (req, res) => {
  try {
    const playlists = await getGeneratedPlaylists(req);
    req.session.generatedPlaylists = playlists.map(serializePlaylist);
    await saveRequestSession(req);

    res.json({
      status: 'success',
      playlists: playlists.map(serializePlaylist),
    });
  } catch (playlistError) {
    console.log('Spotify playlist load error:', formatSpotifyError(playlistError));
    const handledError = getSpotifyRouteError(
      req,
      playlistError,
      'Spotify playlists could not be loaded right now.'
    );

    if (handledError.status === 401 || handledError.status === 403) {
      return res.status(handledError.status).json({
        status: 'error',
        message: handledError.message,
      });
    }

    res.json({
      status: 'success',
      warning: handledError.message,
      playlists: (req.session.generatedPlaylists || []).map(serializePlaylist),
    });
  }
});

app.get('/api/history', authApi, (req, res) => {
  res.json({
    status: 'success',
    playlistCount: (req.session.generatedPlaylists || []).length,
    sessions: (req.session.sessionHistory || []).map(sessionData => ({
      ...sessionData,
      playlist: sessionData.playlist ? serializePlaylist(sessionData.playlist) : null,
    })),
  });
});

app.get('/api/debug/spotify', authApi, async (req, res) => {
  const debugPayload = {
    status: 'success',
    timestamp: new Date().toISOString(),
    redirectUri: process.env.SPOTIFY_REDIRECT_URI,
    grantedScopes: req.session.spotifyGrantedScopes || [],
    missingScopes: getMissingSpotifyScopes(req),
    sessionUser: req.session.spotifyProfile || null,
    checks: {},
  };

  async function runCheck(name, requestFactory, formatResponse) {
    try {
      const response = await requestFactory();
      debugPayload.checks[name] = {
        ok: true,
        status: response.status,
        data: formatResponse ? formatResponse(response) : null,
      };
    } catch (error) {
      debugPayload.checks[name] = {
        ok: false,
        status: error.response?.status || null,
        message: getSpotifyErrorDescription(error),
      };
    }
  }

  await runCheck(
    'profile',
    () =>
      spotifyApiRequest(req, {
        method: 'get',
        url: '/me',
      }),
    response => serializeSpotifyProfile(response.data)
  );

  await runCheck(
    'currentlyPlaying',
    () =>
      spotifyApiRequest(req, {
        method: 'get',
        url: '/me/player/currently-playing',
        validateStatus(status) {
          return (status >= 200 && status < 300) || status === 204;
        },
      }),
    response => ({
      hasActiveTrack: Boolean(response.data?.item),
      playbackType: response.data?.currently_playing_type || null,
      isPlaying: Boolean(response.data?.is_playing),
      trackName: response.data?.item?.name || null,
    })
  );

  await runCheck(
    'recentlyPlayed',
    () =>
      spotifyApiRequest(req, {
        method: 'get',
        url: '/me/player/recently-played',
        params: {
          limit: 1,
        },
      }),
    response => ({
      itemCount: response.data?.items?.length || 0,
      latestTrack: response.data?.items?.[0]?.track?.name || null,
    })
  );

  await runCheck(
    'playlists',
    () =>
      spotifyApiRequest(req, {
        method: 'get',
        url: '/me/playlists',
        params: {
          limit: 1,
        },
      }),
    response => ({
      itemCount: response.data?.items?.length || 0,
    })
  );

  if (debugPayload.checks.profile.ok && debugPayload.checks.profile.data) {
    req.session.spotifyProfile = debugPayload.checks.profile.data;
    await saveRequestSession(req);
  }

  res.json(debugPayload);
});

function startServer() {
  return app.listen(PORT, HOST, () => {
    console.log(`Server is listening on http://${HOST}:${PORT}`);
  });
}

if (require.main === module) {
  startServer();
}

module.exports = {
  app,
  startServer,
};
