import React, { useEffect, useState } from 'react';
import { WordPressCheckResult } from '../types';

interface WordPressCheckerProps {
  url: string;
  onCheckComplete?: (result: WordPressCheckResult | null) => void;
}

const WordPressChecker: React.FC<WordPressCheckerProps> = ({ url, onCheckComplete }) => {
  const [isChecking, setIsChecking] = useState<boolean>(false);
  const [result, setResult] = useState<WordPressCheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Only check if URL is provided and looks valid
    if (!url || !url.trim() || !url.startsWith('http')) {
      setResult(null);
      setError(null);
      setIsChecking(false);
      if (onCheckComplete) onCheckComplete(null);
      return;
    }

    // Debounce the check
    const timeoutId = setTimeout(() => {
      checkWordPress(url);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [url]);

  const checkWordPress = async (siteUrl: string) => {
    setIsChecking(true);
    setError(null);
    setResult(null);

    try {
      // Call backend endpoint - adjust the URL to match your backend
      const response = await fetch('/api/check-wordpress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: siteUrl }),
      });

      if (!response.ok) {
        throw new Error('Failed to check WordPress site');
      }

      const data: WordPressCheckResult = await response.json();
      setResult(data);
      if (onCheckComplete) onCheckComplete(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      setResult(null);
      if (onCheckComplete) onCheckComplete(null);
    } finally {
      setIsChecking(false);
    }
  };

  // Don't render anything if no URL or not checking
  if (!url || !url.trim()) {
    return null;
  }

  return (
    <div className="mt-2 text-sm">
      {isChecking && (
        <div className="text-blue-400 flex items-center gap-2">
          <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Checking WordPress site...
        </div>
      )}
      {error && (
        <div className="text-red-400">
          ⚠️ {error}
        </div>
      )}
      {result && !isChecking && (
        <div className={`flex items-center gap-2 ${result.is_wordpress ? 'text-emerald-400' : 'text-yellow-400'}`}>
          {result.is_wordpress ? (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              WordPress site detected (score: {result.score}/5)
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              WordPress not detected (score: {result.score}/5)
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default WordPressChecker;

