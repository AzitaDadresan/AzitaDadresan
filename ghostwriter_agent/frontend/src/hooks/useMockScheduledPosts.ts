import { useState, useEffect } from 'react';
import { ScheduledPost } from './useScheduledPosts';

/**
 * Hook for fetching scheduled posts from backend API
 */
export const useMockScheduledPosts = (db: any, userId: string | null) => {
  const [scheduledPosts, setScheduledPosts] = useState<ScheduledPost[]>([]);

  useEffect(() => {
    if (!userId) {
      setScheduledPosts([]);
      return;
    }

    // Fetch from backend
    const loadPosts = async () => {
      try {
        const response = await fetch('/api/scheduled-posts/list', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: userId,
          }),
        });

        if (!response.ok) {
          console.error('Failed to fetch scheduled posts');
          setScheduledPosts([]);
          return;
        }

        const data = await response.json();
        const posts = data.posts || [];
        
        setScheduledPosts(posts.sort((a: ScheduledPost, b: ScheduledPost) => 
          new Date(a.dateTime).getTime() - new Date(b.dateTime).getTime()
        ));
      } catch (e) {
        console.error('Error loading scheduled posts:', e);
        setScheduledPosts([]);
      }
    };

    loadPosts();

    // Poll for updates every 5 seconds
    const interval = setInterval(loadPosts, 5000);

    return () => clearInterval(interval);
  }, [userId]);

  return scheduledPosts;
};

