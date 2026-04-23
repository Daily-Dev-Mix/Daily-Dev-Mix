process.env.SESSION_SECRET = 'test-session-secret';
process.env.SPOTIFY_CLIENT_ID = 'spotify-client-id';
process.env.SPOTIFY_CLIENT_SECRET = 'spotify-client-secret';
process.env.SPOTIFY_REDIRECT_URI = 'http://127.0.0.1:3000/auth/spotify/callback';

const chai = require('chai');
const { app } = require('../index');

const { assert, expect } = chai;

function getRouteStack(method, routePath) {
  const layer = app._router.stack.find(entry => {
    return entry.route && entry.route.path === routePath && entry.route.methods[method];
  });

  if (!layer) {
    throw new Error(`Route not found for ${method.toUpperCase()} ${routePath}`);
  }

  return layer.route.stack.map(entry => entry.handle);
}

async function invokeRoute(method, routePath, overrides = {}) {
  const handlers = getRouteStack(method, routePath);

  const req = {
    method: method.toUpperCase(),
    path: routePath,
    url: routePath,
    body: overrides.body || {},
    query: overrides.query || {},
    session: {
      save(callback) {
        if (callback) {
          callback();
        }
      },
      ...(overrides.session || {}),
    },
  };

  const res = {
    statusCode: 200,
    body: null,
    headers: {},
    view: null,
    locals: null,
    finished: false,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      this.finished = true;
      return this;
    },
    redirect(location) {
      if (this.statusCode === 200) {
        this.statusCode = 302;
      }

      this.headers.location = location;
      this.finished = true;
      return this;
    },
    render(view, locals) {
      this.view = view;
      this.locals = locals;
      this.finished = true;
      return this;
    },
  };

  for (const handler of handlers) {
    if (res.finished) {
      break;
    }

    await new Promise((resolve, reject) => {
      let nextCalled = false;

      const next = error => {
        nextCalled = true;

        if (error) {
          reject(error);
          return;
        }

        resolve();
      };

      try {
        const result = handler(req, res, next);

        if (result && typeof result.then === 'function') {
          result
            .then(() => {
              if (!nextCalled) {
                resolve();
              }
            })
            .catch(reject);
          return;
        }

        if (res.finished || handler.length < 3) {
          resolve();
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  return { req, res };
}

describe('Server routes', () => {
  it('Returns the default welcome message', async () => {
    const { res } = await invokeRoute('get', '/welcome');

    expect(res.statusCode).to.equal(200);
    expect(res.body.status).to.equal('success');
    assert.strictEqual(res.body.message, 'Welcome!');
  });

  it('Redirects unauthenticated page requests to /login', async () => {
    const { res } = await invokeRoute('get', '/home');

    expect(res.statusCode).to.equal(302);
    expect(res.headers.location).to.equal('/login');
  });

  it('Rejects unauthenticated API session starts', async () => {
    const { res } = await invokeRoute('post', '/api/session/start', {
      body: { label: 'Studying', emoji: '📚' },
    });

    expect(res.statusCode).to.equal(401);
    expect(res.body.status).to.equal('error');
    assert.strictEqual(res.body.message, 'Please log in with Spotify first.');
  });

  it('Builds a Spotify authorization redirect with the expected scopes', async () => {
    const { req, res } = await invokeRoute('get', '/auth/spotify');

    expect(res.statusCode).to.equal(302);
    expect(res.headers.location).to.include('https://accounts.spotify.com/authorize?');
    expect(res.headers.location).to.include('user-read-private');
    expect(res.headers.location).to.include('user-read-currently-playing');
    expect(res.headers.location).to.include('user-read-recently-played');
    expect(res.headers.location).to.include('playlist-read-private');
    expect(res.headers.location).to.include('playlist-modify-private');
    expect(req.session.spotifyAuthState).to.be.a('string');
    expect(req.session.spotifyAuthState.length).to.be.greaterThan(10);
  });
});
