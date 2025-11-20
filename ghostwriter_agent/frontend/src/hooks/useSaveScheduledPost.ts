import { useCallback } from 'react';
import { Firestore } from 'firebase/firestore';
import { collection, doc, setDoc } from 'firebase/firestore';
import { getFirebaseConfig } from '../constants/generator';

export const useSaveScheduledPost = (db: Firestore | null, userId: string | null) => {
  const saveScheduledPost = useCallback(async (
    platform: string, 
    content: string, 
    date: string, 
    timeKey: string, 
    imageUrl: string | null
  ) => {
    if (!db || !userId) {
      throw new Error("Error: Authentication not ready. Cannot save post.");
    }
    
    const { appId } = getFirebaseConfig();
    const [h, m] = timeKey.split(':');
    const timeFormatted = `${h.padStart(2, '0')}:${m}`;
    const dateTime = `${date}T${timeFormatted}:00`;
    const collectionPath = `/artifacts/${appId}/users/${userId}/scheduled_posts`;
    
    try {
      await setDoc(doc(collection(db, collectionPath)), {
        platform,
        content,
        dateTime,
        status: 'Scheduled',
        imageUrl: imageUrl || null,
        createdAt: new Date().toISOString(),
      });
      return { success: true, message: `Post for ${platform} successfully scheduled! View on Dashboard.` };
    } catch (error: any) {
      console.error("Error saving post to Firestore:", error);
      throw new Error(`Failed to schedule post: ${error.message}`);
    }
  }, [db, userId]);

  return { saveScheduledPost };
};

