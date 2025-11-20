import React, { useState } from 'react';
import { ScheduledPost } from '../hooks/useScheduledPosts';

interface ScheduledPostsDashboardProps {
  posts: ScheduledPost[];
  userId: string | null;
  onPostUpdated?: () => void;
}

const ScheduledPostsDashboard: React.FC<ScheduledPostsDashboardProps> = ({ posts, userId, onPostUpdated }) => {
  const [publishingPostId, setPublishingPostId] = useState<string | null>(null);
  const [publishError, setPublishError] = useState<string | null>(null);
  const [publishSuccess, setPublishSuccess] = useState<string | null>(null);

  const handlePublishToWordPress = async (postId: string, platform: string) => {
    if (!userId) {
      setPublishError("User not authenticated");
      return;
    }

    if (platform.toLowerCase() !== 'wordpress') {
      setPublishError("Only WordPress posts can be published directly");
      return;
    }

    setPublishingPostId(postId);
    setPublishError(null);
    setPublishSuccess(null);

    try {
      const response = await fetch(`/api/scheduled-posts/publish-wordpress/${userId}/${postId}`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to publish to WordPress');
      }

      const data = await response.json();
      setPublishSuccess(data.message || 'Post published successfully!');
      
      // Refresh the posts list
      if (onPostUpdated) {
        onPostUpdated();
      }
    } catch (error: any) {
      setPublishError(error.message || 'Failed to publish to WordPress');
    } finally {
      setPublishingPostId(null);
      
      // Clear messages after 5 seconds
      setTimeout(() => {
        setPublishError(null);
        setPublishSuccess(null);
      }, 5000);
    }
  };

  const handleDeletePost = async (postId: string) => {
    if (!userId || !confirm('Are you sure you want to delete this scheduled post?')) {
      return;
    }

    try {
      const response = await fetch(`/api/scheduled-posts/${userId}/${postId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete post');
      }

      // Refresh the posts list
      if (onPostUpdated) {
        onPostUpdated();
      }
    } catch (error: any) {
      setPublishError(error.message || 'Failed to delete post');
      setTimeout(() => setPublishError(null), 5000);
    }
  };

  return (
    <section className="py-10 px-4 md:px-8">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-white">Scheduled Posts Dashboard</h2>
        
        {/* Status Messages */}
        {publishError && (
          <div className="mb-4 p-4 rounded-lg bg-red-900/50 border border-red-500 text-red-200">
            <p className="font-medium">Error: {publishError}</p>
          </div>
        )}
        
        {publishSuccess && (
          <div className="mb-4 p-4 rounded-lg bg-green-900/50 border border-green-500 text-green-200">
            <p className="font-medium">✓ {publishSuccess}</p>
          </div>
        )}
        
        {posts.length === 0 ? (
          <div className="text-center p-12 rounded-xl bg-slate-800 text-slate-400 border-dashed border-2 border-slate-700">
            <p className="text-xl font-medium">No posts currently scheduled.</p>
            <p className="text-sm mt-2">Generate a draft and click 'Schedule Post' to see it here.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {posts.map(post => (
              <div key={post.id} className="p-5 rounded-xl bg-slate-800 shadow-lg border-l-4 border-emerald-500">
                <div className="flex justify-between items-start flex-wrap gap-4">
                  <div className="flex-1 min-w-[300px]">
                    <div className="flex items-center gap-2 mb-2">
                      <p className="text-lg font-bold text-white">{post.platform} Post</p>
                      {post.status === 'Published' && (
                        <span className="px-2 py-1 text-xs rounded-full bg-green-900/50 text-green-300 border border-green-500">
                          Published
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-400 line-clamp-2">{post.content}</p>
                    {post.imageUrl && (
                      <div className="mt-2 flex items-center text-xs text-blue-400">
                        <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4.586-4.586a2 2 0 012.828 0L14 13.172l-1.586 1.586a2 2 0 01-2.828 0L8 12.828l-2.586 2.586zm4-10a1 1 0 00-1-1H4a1 1 0 00-1 1v1h14V5z" clipRule="evenodd"></path>
                        </svg>
                        Image included
                      </div>
                    )}
                    {(post as any).wordpressUrl && (
                      <a 
                        href={(post as any).wordpressUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="mt-2 inline-flex items-center text-xs text-blue-400 hover:text-blue-300"
                      >
                        <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"></path>
                          <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"></path>
                        </svg>
                        View on WordPress
                      </a>
                    )}
                  </div>
                  <div className="text-right">
                    <p className="text-emerald-400 font-mono text-xl">
                      {new Date(post.dateTime).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: 'numeric' })}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Status: {post.status}</p>
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="mt-4 flex gap-2 flex-wrap">
                  {post.platform.toLowerCase() === 'wordpress' && post.status !== 'Published' && (
                    <button
                      onClick={() => handlePublishToWordPress(post.id, post.platform)}
                      disabled={publishingPostId === post.id}
                      className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors flex items-center gap-2"
                    >
                      {publishingPostId === post.id ? (
                        <>
                          <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Publishing...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 1.414L10.586 9H7a1 1 0 100 2h3.586l-1.293 1.293a1 1 0 101.414 1.414l3-3a1 1 0 000-1.414z" clipRule="evenodd"></path>
                          </svg>
                          Publish to WordPress
                        </>
                      )}
                    </button>
                  )}
                  
                  <button
                    onClick={() => handleDeletePost(post.id)}
                    className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd"></path>
                    </svg>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export default ScheduledPostsDashboard;


