import { useState, useEffect } from 'react';

export interface MockUser {
  userId: string;
  email: string;
  displayName?: string;
}

/**
 * Mock Firebase hook for testing purposes.
 * Simulates authentication without requiring actual Firebase connection.
 */
export const useMockFirebase = () => {
  const [db, setDb] = useState<any>(null);
  const [auth, setAuth] = useState<any>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [user, setUser] = useState<MockUser | null>(null);
  const [isAuthReady, setIsAuthReady] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = () => {
      if (typeof window !== 'undefined') {
        const storedUser = localStorage.getItem('mock_user');
        if (storedUser) {
          try {
            const userData: MockUser = JSON.parse(storedUser);
            setUser(userData);
            setUserId(userData.userId);
            setDb(null);
            setAuth(null);
            setIsAuthReady(true);
            return;
          } catch (e) {
            console.error('Error parsing stored user:', e);
          }
        }
      }
      // No existing session
      setIsAuthReady(true);
    };

    const initTimer = setTimeout(checkAuth, 300);
    return () => clearTimeout(initTimer);
  }, []);

  const signIn = async (email: string, password: string): Promise<MockUser> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Mock authentication - in real app, this would validate with Firebase
    // For demo, accept any email/password
    if (!email || !password) {
      throw new Error('Email and password are required');
    }

    const mockUser: MockUser = {
      userId: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      email: email,
      displayName: email.split('@')[0],
    };

    if (typeof window !== 'undefined') {
      localStorage.setItem('mock_user', JSON.stringify(mockUser));
      localStorage.setItem('mock_user_id', mockUser.userId);
    }

    setUser(mockUser);
    setUserId(mockUser.userId);
    setDb(null);
    setAuth(null);
    setError(null);

    return mockUser;
  };

  const signUp = async (email: string, password: string, displayName?: string): Promise<MockUser> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    if (!email || !password) {
      throw new Error('Email and password are required');
    }

    if (password.length < 6) {
      throw new Error('Password must be at least 6 characters');
    }

    const mockUser: MockUser = {
      userId: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      email: email,
      displayName: displayName || email.split('@')[0],
    };

    if (typeof window !== 'undefined') {
      localStorage.setItem('mock_user', JSON.stringify(mockUser));
      localStorage.setItem('mock_user_id', mockUser.userId);
    }

    setUser(mockUser);
    setUserId(mockUser.userId);
    setDb(null);
    setAuth(null);
    setError(null);

    return mockUser;
  };

  const signOut = async (): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 200));
    
    if (typeof window !== 'undefined') {
      localStorage.removeItem('mock_user');
      localStorage.removeItem('mock_user_id');
      localStorage.removeItem('mock_session_id');
      localStorage.removeItem('mock_auth_token');
    }

    setUser(null);
    setUserId(null);
    setDb(null);
    setAuth(null);
    setError(null);
  };

  return { 
    db, 
    auth, 
    userId,
    user,
    isAuthReady, 
    error,
    signIn,
    signUp,
    signOut,
  };
};
