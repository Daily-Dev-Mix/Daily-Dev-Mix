CREATE TABLE if not exists users (
    id SERIAL PRIMARY KEY,
    spotify_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    username VARCHAR(50) PRIMARY KEY,
    password VARCHAR(60) NOT NULL
);

CREATE TABLE if not exists sessions(
    session_id SERIAL PRIMARY KEY,
    user_id INT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE sessions
ADD CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users (id);

CREATE TABLE if not exists songs(
    id SERIAL PRIMARY KEY,
    spotify_song_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    artist VARCHAR(100) NOT NULL,
    album VARCHAR(100) NOT NULL,
    image_url TEXT,
    duration INT NOT NULL
);

CREATE TABLE if not exists songs_to_sessions(
    id SERIAL PRIMARY KEY,
    session_id INT,
    song_id INT,
    time_played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE songs_to_sessions
ADD CONSTRAINT fk_session_id FOREIGN KEY (session_id) REFERENCES sessions (session_id);
ALTER TABLE songs_to_sessions
ADD CONSTRAINT fk_song_id FOREIGN KEY (song_id) REFERENCES songs (id);