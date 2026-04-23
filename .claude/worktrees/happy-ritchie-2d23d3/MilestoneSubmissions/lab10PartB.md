# User Acceptance Test Plan
## Daily Dev Mix - CSCI 3308

---

## Test Environment
All tests will be conducted on localhost using Docker Compose.

## Testers
CU Boulder CSCI 3308 students acting as target users of the Daily Dev Mix application.

---

## Feature 1: User Registration

### Description
A new user can create an account by providing their name, email, username, and password.

### Test Cases

#### Test Case 1.1 - Positive: Valid Registration
- **Test Data:** name: "John Doe", email: "johndoe@example.com", username: "johndoe", password: "Password1"
- **Steps:** Navigate to /register, fill in all fields with valid data, click Register
- **Expected Result:** User is redirected to the login page and account is created in the database
- **Actual Result:** 

#### Test Case 1.2 - Negative: Missing Fields
- **Test Data:** name: "", email: "", username: "", password: ""
- **Steps:** Navigate to /register, leave all fields blank, click Register
- **Expected Result:** Form does not submit, user sees an error message
- **Actual Result:** 

---

## Feature 2: Login with Spotify

### Description
A registered user can log in using their Spotify account via OAuth, connecting their Spotify data to the app.

### Test Cases

#### Test Case 2.1 - Positive: Valid Spotify Login
- **Test Data:** Valid Spotify account credentials
- **Steps:** Navigate to /login, click "Connect with Spotify", authorize the app on Spotify's login page
- **Expected Result:** User is redirected back to the app and logged in successfully
- **Actual Result:** 

#### Test Case 2.2 - Negative: Deny Spotify Authorization
- **Test Data:** Valid Spotify account credentials
- **Steps:** Navigate to /login, click "Connect with Spotify", deny authorization on Spotify's login page
- **Expected Result:** User is redirected back to the app with an error message and is not logged in
- **Actual Result:** 

---

## Feature 3: Logout

### Description
A logged in user can log out of the application, ending their session.

### Test Cases

#### Test Case 3.1 - Positive: Successful Logout
- **Test Data:** A currently logged in user session
- **Steps:** Click the logout button while logged in
- **Expected Result:** Session is cleared, user is redirected to the login page and cannot access authenticated routes
- **Actual Result:** 

#### Test Case 3.2 - Negative: Accessing Authenticated Route After Logout
- **Test Data:** A previously logged in session that has been logged out
- **Steps:** Log out, then attempt to navigate directly to /home
- **Expected Result:** User is redirected to /login and cannot access the page
-
