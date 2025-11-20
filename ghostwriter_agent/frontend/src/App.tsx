import './App.css'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useState } from 'react'
import Generator from './components/Generator'
import LandingPage from './components/LandingPage'
import Login from './components/Login'
import SignUp from './components/SignUp'
import ProtectedRoute from './components/ProtectedRoute'
import PublicRoute from './components/PublicRoute'
import { useFirebase } from './hooks/useFirebase'

function App() {
  const { isAuthReady, signIn, signUp, signOut, error: authError } = useFirebase();
  const [authErrorState, setAuthErrorState] = useState<string | null>(null);

  const handleLogin = async (email: string, password: string) => {
    setAuthErrorState(null);
    try {
      await signIn(email, password);
    } catch (err: any) {
      setAuthErrorState(err.message);
      throw err;
    }
  };

  const handleSignUp = async (email: string, password: string, displayName?: string) => {
    setAuthErrorState(null);
    try {
      await signUp(email, password, displayName);
    } catch (err: any) {
      setAuthErrorState(err.message);
      throw err;
    }
  };

  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <p className="text-xl text-blue-300">Initializing...</p>
      </div>
    );
  }

  return (
    <Routes>
      {/* Public Routes */}
      <Route 
        path="/" 
        element={
          <PublicRoute>
            <LandingPage />
          </PublicRoute>
        } 
      />
      <Route 
        path="/login" 
        element={
          <PublicRoute>
            <Login 
              onLogin={handleLogin} 
              onSwitchToSignUp={() => {}}
              error={authErrorState || authError} 
            />
          </PublicRoute>
        } 
      />
      <Route 
        path="/signup" 
        element={
          <PublicRoute>
            <SignUp 
              onSignUp={handleSignUp} 
              onSwitchToLogin={() => {}}
              error={authErrorState || authError} 
            />
          </PublicRoute>
        } 
      />

      {/* Protected Routes */}
      <Route 
        path="/generator" 
        element={
          <ProtectedRoute>
            <Generator onSignOut={signOut} />
          </ProtectedRoute>
        } 
      />

      {/* Catch all - redirect to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App

