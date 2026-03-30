CREATE TABLE if not exists users (
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    username VARCHAR(50) PRIMARY KEY,
    password VARCHAR(60) NOT NULL
);