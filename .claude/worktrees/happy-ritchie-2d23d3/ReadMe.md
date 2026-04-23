## Daily Dev Mix

### Application Description: 
Daily Dev Mix is a Spotify companion tool whose main goal is to create personalized playlists for the user based on activities they do while listening to music. The user signs into the website using their Spotify account before being redirected to the dashboard. The user is then able to select a predefined or user-created “vibe” or activity they will be doing as they start their listening session. Once the “vibe” is selected and the start session button is clicked, the user can start listening to music on Spotify, and their listening data will be displayed and collected on the site using Spotify API calls. After the session, a playlist is created based on the music that was listened to during that activity. These playlists are built on with each successive listening session and are separated based on the different activities. Spotify does have special activity playlists built in, but our API aims to create personalized playlists based on the music listened to during the session, so users can enjoy music they actually want to hear while engaging in their activity.

### Contributers: 
- Devang Pandey (DevangPandey1)
- Ian Martin (IanMartin110)
- Matthew Aldridge (Matthew-Aldridge)
- Greg Lowman (GregLowman)
- Dylan Long (dylo3261)
- Vera Zaric (vera-z05)

### Technology Stack
- Front End: HTML Handlebars
- Database: PostgreSQL
- Functionality: JavaScript and Express
- Application Server: NodeJS
- External API: Spotify
- Testing tool: Mocha

### Prerequisites to run the application:
To use the application a valid Spotify account is needed to login.
To run the application locally Docker is needed.

### Instructions on how to run the application locally:
1. Navigate to the ProjectSourceCode directory in the repository
2. Run the command 'docker-compose up --build' to initialize the docker container and start the application

### How to run the tests:
Tests for the application are ran when running 'docker-compose up' or using npm test

### Link to the deployed application:
https://daily-dev-mix-production.up.railway.app
