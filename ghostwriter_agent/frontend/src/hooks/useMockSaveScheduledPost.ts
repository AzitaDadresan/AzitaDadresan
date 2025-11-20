import { useCallback } from 'react';
import { ScheduledPost } from './useScheduledPosts';

/**
 * Hook for saving scheduled posts to backend API
 */
export const useMockSaveScheduledPost = (db: any, userId: string | null) => {
  const saveScheduledPost = useCallback(async (
    platform: string, 
    content: string, 
    date: string, 
    timeKey: string, 
    imageUrl: string | null
  ) => {
    if (!userId) {
      throw new Error("Error: User not authenticated. Cannot save post.");
    }
    
    const [h, m] = timeKey.split(':');
    const timeFormatted = `${h.padStart(2, '0')}:${m}`;
    const dateTime = `${date}T${timeFormatted}:00`;

    try {
      // Call backend API to save scheduled post
      const response = await fetch('/api/scheduled-posts/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          platform: platform,
          content: content,
          date_time: dateTime,
          image_url: imageUrl,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save scheduled post');
      }

      const data = await response.json();
      
      return { 
        success: true, 
        message: data.message || `Post for ${platform} successfully scheduled! View on Dashboard.` 
      };
    } catch (error: any) {
      console.error("Error saving scheduled post:", error);
      throw new Error(`Failed to schedule post: ${error.message}`);
    }
  }, [userId]);

  return { saveScheduledPost };
};

