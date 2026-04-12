require('dotenv').config();

// Import Dependencies -->
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

//START OF API ROUTES
app.get('/', (req, res) => {
  res.redirect('/login');
});

app.get('/login', (req, res) => {
    res.render('pages/login');
});

//Send Spotify data

app.get('/auth/spotify', (req, res) => {
  const scope = 'user-read-currently-playing user-read-playback-state playlist-modify-public playlist-modify-private';
  
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: process.env.SPOTIFY_CLIENT_ID,
    scope: scope,
    redirect_uri: process.env.SPOTIFY_REDIRECT_URI,
  });

  res.redirect('https://accounts.spotify.com/authorize?' + params);
});

//Recieve returned data from spotify

app.get('/auth/spotify/callback', async (req, res) => {
  const code = req.query.code;

  try {
    const response = await axios.post(
      'https://accounts.spotify.com/api/token',
      new URLSearchParams({
        grant_type: 'authorization_code',
        code: code,
        redirect_uri: process.env.SPOTIFY_REDIRECT_URI,
      }),
      {
        headers: {
          'Authorization': 'Basic ' + Buffer.from(
            process.env.SPOTIFY_CLIENT_ID + ':' + process.env.SPOTIFY_CLIENT_SECRET
          ).toString('base64'),
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      }
    );

    req.session.spotifyToken = response.data.access_token;
    req.session.user = { authenticated: true };

    req.session.save(() => {
      res.redirect('/home');
    });

  } catch (error) {
    console.log('Spotify auth error:', error.message);
    res.redirect('/login');
  }
});
// Authentication Middleware: require user to be logged in
const auth = (req, res, next) => {
  if (!req.session.user) {
    // Default to login page.
    return res.redirect('/login');
  }
  next();
};
app.get('/home', auth, (req, res) => {
  res.render('pages/home', { activePage: 'home' });
});

//Welcome route for lab 10
app.get('/welcome', (req, res) => {
  res.json({status: 'success', message: 'Welcome!'});
});

app.get('/playlists', auth, (req, res) => {
  res.render('pages/playlists', { activePage: 'playlists' });
})

app.get('/history', auth, (req, res) => {
  res.render("pages/history", { activePage: 'history' });
});

app.get('/active-session', auth, (req, res) => {
  res.render('pages/active-session');
});

//This get route is used to get the current song data from spotify. It will return null if no song.
//This needs to be linked to frontend page and called frequently or have a way to call it every time the song changes
app.get('/api/active-session', auth, async (req, res) => {
  try {
    const token = req.session.spotifyToken;

    if (!token) {
      return res.status(401).json({ error: 'No Spotify token' });
    }
    const nowPlayingRes = await axios.get(
      'https://api.spotify.com/v1/me/player/currently-playing',
      {
        headers: {
          Authorization: `Bearer ${token}`
        }
      }
    );
    const nowPlaying = nowPlayingRes.data;
    let currentTrack = null;
    if (nowPlaying.item) {
      const track = nowPlaying.item;
      currentTrack = {
        id: track.id,
        name: track.name,
        artist: track.artists.map(a => a.name).join(', '),
        album: track.album.name,
        imageUrl: track.album.images?.[0]?.url,
        duration: Math.floor(track.duration_ms / 1000)
      };
      console.log('Current track:', currentTrack);
    }
    res.json({
      nowPlaying: currentTrack
    });
  } catch (err) {
    console.error('Session API error:', err.message);
    res.status(500).json({ error: 'Failed to load session data' });
  }
});


// starting the server and keeping the connection open to listen for more requests
module.exports = app.listen(PORT, HOST, () => {
  console.log(`Server is listening on http://${HOST}:${PORT}`);
});