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
    secret: process.env.SESSION_SECRET,
    saveUninitialized: false,
    resave: false,
  })
);
app.use(
  bodyParser.urlencoded({
    extended: true,
  })
);

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
    // Fetch Spotify profile
    const accessToken = response.data.access_token;
    const profileResponse = await axios.get('https://api.spotify.com/v1/me', {
      headers: { Authorization: 'Bearer ' + accessToken },
    });
    req.session.spotifyToken = accessToken;
    req.session.user = {
      authenticated: true,
      name: profileResponse.data.display_name,
      image: profileResponse.data.images?.[0]?.url,
      spotifyId: profileResponse.data.id,
    };
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
  res.render('pages/home', {
    user: req.session.user,
  });
});
//Welcome route for lab 10
app.get('/welcome', (req, res) => {
  res.json({status: 'success', message: 'Welcome!'});
});

app.get('/playlists', auth, (req, res) => {
  res.render('pages/playlists');
})

app.get('/history', auth, (req, res) => {
  res.render('pages/history');
});

app.get('/active-session', auth, (req, res) => {
  res.render('pages/active-session');
});



// starting the server and keeping the connection open to listen for more requests
module.exports = app.listen(3000, '0.0.0.0', () => {
  console.log('Server is listening on port 3000');
});