# Firebase Setup Guide for GhostWriter

## 1. Create a Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or select an existing project
3. Follow the setup wizard (you can disable Google Analytics if not needed)

## 2. Enable Authentication

1. In your Firebase project, go to **Build** → **Authentication**
2. Click "Get started"
3. Go to the **Sign-in method** tab
4. Enable **Email/Password** provider
5. Click "Save"

## 3. Create a Web App

1. In Project Settings (gear icon), scroll down to "Your apps"
2. Click the web icon `</>` to add a web app
3. Register your app with a nickname (e.g., "GhostWriter Web")
4. Copy the Firebase configuration object

## 4. Get Your Configuration

You'll see something like this:

```javascript
const firebaseConfig = {
  apiKey: "AIzaSyC7X8h...",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789012",
  appId: "1:123456789012:web:abc123def456"
};
```

## 5. Configure Your App

1. Create a `.env` file in the `frontend/` directory:
   ```bash
   cd frontend
   cp .env.example .env
   ```

2. Edit `.env` and add your Firebase config values:
   ```env
   VITE_FIREBASE_API_KEY=AIzaSyC7X8h...
   VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
   VITE_FIREBASE_PROJECT_ID=your-project
   VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
   VITE_FIREBASE_MESSAGING_SENDER_ID=123456789012
   VITE_FIREBASE_APP_ID=1:123456789012:web:abc123def456
   ```

## 6. (Optional) Set Up Firestore Database

If you want to use Firestore for scheduled posts storage:

1. Go to **Build** → **Firestore Database**
2. Click "Create database"
3. Start in **test mode** for development
4. Choose a location closest to your users
5. Click "Enable"

### Security Rules for Development

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow authenticated users to read/write their own data
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow authenticated users to read/write scheduled posts
    match /scheduled_posts/{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

## 7. Test Your Setup

1. Start the frontend:
   ```bash
   npm run dev
   ```

2. Try to sign up with a new email and password
3. You should see the new user in Firebase Console → Authentication → Users

## Troubleshooting

### "Firebase: Error (auth/operation-not-allowed)"
- Make sure Email/Password authentication is enabled in Firebase Console

### "Firebase: API key not valid"
- Check that your `VITE_FIREBASE_API_KEY` matches exactly what's in Firebase Console
- Make sure there are no extra spaces or quotes

### "Firebase: Firebase App named '[DEFAULT]' already exists"
- Restart your dev server (`npm run dev`)
- Clear browser cache and local storage

### Network Errors
- Check your internet connection
- Verify Firebase project is not in a restricted region
- Check browser console for CORS errors

## Security Notes

- **Never commit `.env` to git** - it's already in `.gitignore`
- Use Firebase Security Rules to protect your database
- In production, restrict API key usage to your domain
- Consider using Firebase App Check for additional security

## Next Steps

- Set up Firestore Security Rules for production
- Enable additional auth providers (Google, GitHub, etc.)
- Add email verification for new users
- Implement password reset functionality
