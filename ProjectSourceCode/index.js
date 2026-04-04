// Import Dependencies -->
const express = require('express'); // To build an application server or API
const app = express();
const handlebars = require('express-handlebars'); //to enable express to work with handlebars
const Handlebars = require('handlebars'); // to include the templating engine responsible for compiling templates
const path = require('path');
const pgp = require('pg-promise')(); // To connect to the Postgres DB from the node server
const bodyParser = require('body-parser');
const session = require('express-session'); // To set the session object. To store or access session data, use the `req.session`, which is (generally) serialized as JSON by the store.
const bcrypt = require('bcryptjs'); //  To hash passwords
const axios = require('axios'); // To make HTTP requests from our server. We'll learn more about it in Part C.

//Connect to DB -->
// create `ExpressHandlebars` instance and configure the layouts and partials dir.
const hbs = handlebars.create({
  extname: 'hbs',
  layoutsDir: __dirname + '/views/layouts',
  partialsDir: __dirname + '/views/partials',
});
// database configuration
const dbConfig = {
  host: 'db', // the database server
  port: 5432, // the database port
  database: process.env.POSTGRES_DB, // the database name
  user: process.env.POSTGRES_USER, // the user account to connect with
  password: process.env.POSTGRES_PASSWORD, // the password of the user account
};
const db = pgp(dbConfig);
// test database
db.connect()
  .then(obj => {
    console.log('Database connection successful'); // you can view this message in the docker compose logs
    obj.done(); // success, release the connection;
  })
  .catch(error => {
    console.log('ERROR:', error.message || error);
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

Handlebars.registerHelper('eq', (a, b) => a === b);

//START OF API ROUTES
app.get('/', (req, res) => {
  res.redirect('/login');
});
app.get('/register', (req, res) => {
  res.render('pages/register');
});
// Register
app.post('/register', async (req, res) => {
  //hash the password using bcrypt library
  const hash = await bcrypt.hash(req.body.password, 10);
    const query = 'INSERT INTO users(name,email,username,password) VALUES($1, $2,$3,$4)';
    db.none(query,[req.body.name, req.body.email, req.body.username, hash])
    .then(() => {
        res.redirect('/login'); //may want to change to just login automatically after registration
    })
    .catch(error => {
        console.log('ERROR:', error.message || error);
        res.render('pages/register', { message: 'Error registering user', error: 'Error registering user' });
    });
});
app.get('/login', (req, res) => {
    res.render('pages/login');
});
app.post('/login', (req, res) =>{
    const query = 'SELECT * from users where username= $1 or email = $1';
    db.oneOrNone(query, [req.body.identifier])
    .then(async (user)=> {
        if (!user) {
            return res.redirect('/register');
        }
        const match = await bcrypt.compare(req.body.password, user.password);
        if (match) {
            req.session.user = user;
            req.session.save(() => {
                res.redirect('/home');
            });
        }else{
            console.log('ERROR: Invalid username or password');
            res.render('pages/login', { message: 'Invalid username or password', error: 'Invalid username or password' });
        }
    })
    .catch(error => {
        console.log('ERROR:', error.message || error);
        res.render('pages/login', { message: 'Error logging in', error: 'Error logging in' });
    });
})
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

// starting the server and keeping the connection open to listen for more requests
const server = app.listen(3000, () => {
  console.log('Server is listening on port 3000');
});
module.exports = app; 