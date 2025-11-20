import { useState, useEffect } from 'react';
import { 
  Firestore,
  collection, 
  query, 
  onSnapshot 
} from 'firebase/firestore';
import { getFirebaseConfig } from '../constants/generator';

export interface ScheduledPost {
  id: string;
  platform: string;
  content: string;
  dateTime: string;
  status: string;
  imageUrl?: string | null;
  createdAt: string;
}

export const useScheduledPosts = (db: Firestore | null, userId: string | null) => {
  const [scheduledPosts, setScheduledPosts] = useState<ScheduledPost[]>([]);

  useEffect(() => {
    if (!db || !userId) return;

    const { appId } = getFirebaseConfig();
    const collectionPath = `/artifacts/${appId}/users/${userId}/scheduled_posts`;
    const q = query(collection(db, collectionPath));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const posts = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })) as ScheduledPost[];
      // Sort posts by date/time ascending
      setScheduledPosts(posts.sort((a, b) => new Date(a.dateTime).getTime() - new Date(b.dateTime).getTime()));
    }, (error) => {
      console.error("Firestore Snapshot Error:", error);
    });

    return () => unsubscribe();
  }, [db, userId]);

  return scheduledPosts;
};

