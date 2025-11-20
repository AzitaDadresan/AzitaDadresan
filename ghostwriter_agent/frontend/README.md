# Ghostwriter Frontend - Mock Implementation Guide

This README documents all mocked functionality in the frontend and provides a guide for migrating to real services.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Mocked Components](#mocked-components)
- [Migration Guide](#migration-guide)
- [File Locations](#file-locations)
- [Backend Integration](#backend-integration)

---

## 🎯 Overview

The frontend currently uses **mock implementations** for:
- **Authentication** (Firebase)
- **Content Generation** (AI/LLM)
- **Image Generation** (Nanobanana)
- **Scheduled Posts Storage** (Firestore)

These mocks allow the application to run and be tested without requiring actual service credentials or API keys.

---

## 🔧 Mocked Components

### 1. **Authentication (Firebase)**

**Mock File:** `src/hooks/useMockFirebase.ts`

**What's Mocked:**
- User sign-in/sign-up (accepts any email/password)
- User session management
- User data storage (localStorage instead of Firebase Auth)
- Authentication state

**Current Implementation:**
- Stores user data in `localStorage` with keys:
  - `mock_user` - Full user object
  - `mock_user_id` - User ID
- Generates random user IDs on signup
- No actual Firebase connection

**Real Implementation File:** `src/hooks/useFirebase.ts` (already exists, not currently used)

**Where It's Used:**
- `src/App.tsx` - Main app routing
- `src/components/Generator.tsx` - User authentication check
- `src/components/ProtectedRoute.tsx` - Route protection
- `src/components/PublicRoute.tsx` - Public route protection

---

### 2. **Scheduled Posts Storage (Firestore)**

**Mock Files:**
- `src/hooks/useMockScheduledPosts.ts` - Fetching scheduled posts
- `src/hooks/useMockSaveScheduledPost.ts` - Saving scheduled posts

**What's Mocked:**
- Firestore database operations
- Real-time post updates
- Post persistence

**Current Implementation:**
- Stores posts in `localStorage` with key: `mock_scheduled_posts_{userId}`
- Polls localStorage every 1 second to simulate real-time updates
- Posts are stored as JSON arrays

**Real Implementation Files:**
- `src/hooks/useScheduledPosts.ts` - Real Firestore listener
- `src/hooks/useSaveScheduledPost.ts` - Real Firestore save

**Where It's Used:**
- `src/components/Generator.tsx` - Displaying and saving scheduled posts
- `src/components/ScheduledPostsDashboard.tsx` - Dashboard view

---

### 3. **Content Generation (AI/LLM)**

**Mock File:** `src/hooks/useContentGeneration.ts`

**What's Mocked:**
- AI-powered content generation
- Master content creation
- Platform-specific content (LinkedIn, WordPress, Instagram)

**Current Implementation:**
- Generates template-based content using string interpolation
- Simulates 2-second API delay
- Creates mock content for:
  - Master draft (markdown format)
  - LinkedIn (with hashtags)
  - WordPress (HTML format)
  - Instagram (with emojis and hashtags)

**Backend Endpoint Available:**
- `POST /api/run-agent` - Runs the full Ghostwriter agent pipeline
- `POST /api/run-trend-watcher` - Trend discovery agent
- `POST /api/run-content-strategist` - Content strategy agent
- `POST /api/run-content-creator` - Content creation agent

**Where It's Used:**
- `src/components/Generator.tsx` - Main content generation
- `src/components/MasterContentGenerator.tsx` - Generation trigger

---

### 4. **Image Generation (Nanobanana)**

**Mock File:** `src/components/ContentOutput.tsx` (lines 120-137)

**What's Mocked:**
- Nanobanana API image generation
- Image URL generation

**Current Implementation:**
- Uses `placehold.co` placeholder images
- Simulates 2.5-second generation delay
- Creates placeholder URLs with metadata in the text

**Backend Endpoint Available:**
- `POST /api/generate-image` - Generates images using Nanobanana API

**Where It's Used:**
- `src/components/ContentOutput.tsx` - `handleGenerateImage()` function
- `src/components/ImageControlPanel.tsx` - Image generation controls

---

### 5. **WordPress Checker**

**Status:** ✅ **Already Using Real Backend**

**File:** `src/components/WordPressChecker.tsx`

**Implementation:**
- Calls `/api/check-wordpress` endpoint
- Uses real backend service (`helpers/wordpress_checker.py`)

**No Migration Needed** - This is already integrated with the backend.

---

## 🔄 Migration Guide

### Step 1: Replace Mock Authentication with Real Firebase

**Files to Update:**

1. **`src/App.tsx`**
   ```tsx
   // Change from:
   import { useMockFirebase } from './hooks/useMockFirebase';
   const { userId, isAuthReady, signIn, signUp, signOut } = useMockFirebase();
   
   // To:
   import { useFirebase } from './hooks/useFirebase';
   const { userId, isAuthReady, signIn, signUp, signOut } = useFirebase();
   ```

2. **`src/components/Generator.tsx`**
   ```tsx
   // Change from:
   import { useMockFirebase } from '../hooks/useMockFirebase';
   
   // To:
   import { useFirebase } from '../hooks/useFirebase';
   ```

3. **`src/components/ProtectedRoute.tsx`** and **`src/components/PublicRoute.tsx`**
   ```tsx
   // Change from:
   import { useMockFirebase } from '../hooks/useMockFirebase';
   
   // To:
   import { useFirebase } from '../hooks/useFirebase';
   ```

4. **Configure Firebase:**
   - Update `src/constants/generator.ts` - `getFirebaseConfig()` function
   - Add Firebase config to your environment or window object
   - Install Firebase SDK: `npm install firebase`

---

### Step 2: Replace Mock Scheduled Posts with Real Firestore

**Files to Update:**

1. **`src/components/Generator.tsx`**
   ```tsx
   // Change from:
   import { useMockScheduledPosts } from '../hooks/useMockScheduledPosts';
   import { useMockSaveScheduledPost } from '../hooks/useMockSaveScheduledPost';
   const scheduledPosts = useMockScheduledPosts(db, userId);
   const { saveScheduledPost } = useMockSaveScheduledPost(db, userId);
   
   // To:
   import { useScheduledPosts } from '../hooks/useScheduledPosts';
   import { useSaveScheduledPost } from '../hooks/useSaveScheduledPost';
   const scheduledPosts = useScheduledPosts(db, userId);
   const { saveScheduledPost } = useSaveScheduledPost(db, userId);
   ```

**Note:** The real hooks require:
- A valid Firestore `db` instance (from `useFirebase()`)
- A valid `userId` (from authenticated user)
- Firebase configuration in `src/constants/generator.ts`

---

### Step 3: Replace Mock Content Generation with Real Backend API

**File to Update:** `src/hooks/useContentGeneration.ts`

**Current Mock Implementation (lines 21-50):**
```tsx
const generateContent = async (): Promise<string> => {
  // Simulates API call with setTimeout
  await new Promise(resolve => setTimeout(resolve, 2000));
  // Generates template content
}
```

**Replace with Real API Call:**
```tsx
const generateContent = async (): Promise<string> => {
  if (!topic.trim()) {
    throw new Error('Please enter a topic to begin content generation.');
  }

  setIsGenerating(true);

  try {
    const response = await fetch('/api/run-agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        topic: topic,
        prompt: tone // or additional prompt data
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to generate content');
    }

    const data = await response.json();
    
    // Parse response based on your backend structure
    const outputs: ContentOutputs = {
      master: data.master || data.content?.master || '',
      linkedin: data.linkedin || data.content?.linkedin || '',
      wordpress: data.wordpress || data.content?.wordpress || '',
      instagram: data.instagram || data.content?.instagram || '',
    };

    setContentOutputs(outputs);
    setIsGenerating(false);
    return `Master Drafts processed successfully for topic: "${topic}"!`;
  } catch (error: any) {
    setIsGenerating(false);
    throw error;
  }
};
```

**Backend Endpoints Available:**
- `POST /api/run-agent` - Full agent pipeline
- `POST /api/run-content-creator` - Content creation only
- See `backend/api/endpoints.py` for full list

---

### Step 4: Replace Mock Image Generation with Real Nanobanana API

**File to Update:** `src/components/ContentOutput.tsx` (lines 120-137)

**Current Mock Implementation:**
```tsx
const handleGenerateImage = async () => {
  await new Promise(resolve => setTimeout(resolve, 2500));
  const generatedUrl = `https://placehold.co/...`; // Placeholder
}
```

**Replace with Real API Call:**
```tsx
const handleGenerateImage = async () => {
  if (!imagePrompt.trim()) {
    setGlobalAlert('Please enter a descriptive Image Generation Prompt.');
    return;
  }

  setIsGeneratingImage(true);
  handleImageSettingChange(platformKey, 'generatedUrl', null);

  try {
    const response = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: imagePrompt,
        style: style,
        aspect_ratio: aspectRatio,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to generate image');
    }

    const data = await response.json();
    const generatedUrl = data.image_url || data.url;

    handleImageSettingChange(platformKey, 'generatedUrl', generatedUrl);
    setIsGeneratingImage(false);
    setGlobalAlert(`New image visual generated for ${title}!`);
  } catch (error: any) {
    setIsGeneratingImage(false);
    setGlobalAlert(`Error generating image: ${error.message}`);
  }
};
```

**Backend Endpoint:** `POST /api/generate-image`
- See `backend/api/endpoints.py` and `backend/services/image_generator.py`

---

## 📁 File Locations

### Mock Files (To Replace)
```
frontend/src/
├── hooks/
│   ├── useMockFirebase.ts          → Replace with useFirebase.ts
│   ├── useMockScheduledPosts.ts    → Replace with useScheduledPosts.ts
│   └── useMockSaveScheduledPost.ts → Replace with useSaveScheduledPost.ts
└── hooks/
    └── useContentGeneration.ts     → Update generateContent() function
└── components/
    └── ContentOutput.tsx            → Update handleGenerateImage() function
```

### Real Implementation Files (Already Exist)
```
frontend/src/
├── hooks/
│   ├── useFirebase.ts              ✅ Real Firebase implementation
│   ├── useScheduledPosts.ts         ✅ Real Firestore listener
│   └── useSaveScheduledPost.ts      ✅ Real Firestore save
└── constants/
    └── generator.ts                 ✅ Firebase config helper
```

### Components Using Mocks
```
frontend/src/
├── App.tsx                          → Uses useMockFirebase
├── components/
│   ├── Generator.tsx                → Uses all mock hooks
│   ├── ProtectedRoute.tsx           → Uses useMockFirebase
│   └── PublicRoute.tsx              → Uses useMockFirebase
```

---

## 🔌 Backend Integration

### Available Backend Endpoints

The backend API is available at `http://localhost:8000` (or configured port).

**Endpoints:**
- `POST /api/check-wordpress` - ✅ Already integrated
- `POST /api/generate-image` - ⚠️ Needs integration
- `POST /api/run-agent` - ⚠️ Needs integration
- `POST /api/run-trend-watcher` - ⚠️ Optional
- `POST /api/run-content-strategist` - ⚠️ Optional
- `POST /api/run-content-creator` - ⚠️ Optional

**Vite Proxy Configuration:**
The frontend is configured to proxy `/api/*` requests to the backend via `vite.config.ts`:
```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

---

## 🚀 Quick Migration Checklist

- [ ] **Authentication**
  - [ ] Replace `useMockFirebase` with `useFirebase` in all files
  - [ ] Configure Firebase in `src/constants/generator.ts`
  - [ ] Install Firebase SDK: `npm install firebase`
  - [ ] Test sign-in/sign-up flow

- [ ] **Scheduled Posts**
  - [ ] Replace `useMockScheduledPosts` with `useScheduledPosts`
  - [ ] Replace `useMockSaveScheduledPost` with `useSaveScheduledPost`
  - [ ] Verify Firestore collection path in hooks
  - [ ] Test post creation and retrieval

- [ ] **Content Generation**
  - [ ] Update `useContentGeneration.ts` to call `/api/run-agent`
  - [ ] Parse backend response format
  - [ ] Handle errors appropriately
  - [ ] Test content generation flow

- [ ] **Image Generation**
  - [ ] Update `ContentOutput.tsx` to call `/api/generate-image`
  - [ ] Handle image URL response
  - [ ] Test image generation for all platforms

- [ ] **Testing**
  - [ ] Remove all `localStorage` mock data
  - [ ] Test complete user flow
  - [ ] Verify data persistence
  - [ ] Check error handling

---

## 📝 Notes

- **localStorage Keys Used by Mocks:**
  - `mock_user` - User authentication data
  - `mock_user_id` - User ID
  - `mock_scheduled_posts_{userId}` - Scheduled posts per user

- **Cleanup After Migration:**
  - Remove mock hook files (or keep for testing)
  - Clear localStorage mock data
  - Update imports across all components

- **Backend Requirements:**
  - Backend server must be running on port 8000 (or configured port)
  - Environment variables must be set (see root `README.md`)
  - API endpoints must be accessible

---

## 🐛 Troubleshooting

**Issue: Mock data persists after migration**
- Clear browser localStorage
- Remove all `mock_*` keys manually

**Issue: Backend API calls fail**
- Verify backend server is running
- Check Vite proxy configuration
- Verify CORS settings in backend

**Issue: Firebase not initializing**
- Check Firebase config in `src/constants/generator.ts`
- Verify Firebase credentials
- Check browser console for errors

---

For backend API documentation, see: `http://localhost:8000/docs` (FastAPI Swagger UI)

