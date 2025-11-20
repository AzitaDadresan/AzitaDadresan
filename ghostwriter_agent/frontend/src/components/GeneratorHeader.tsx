import React from 'react';
import { useNavigate } from 'react-router-dom';

interface GeneratorHeaderProps {
  userId: string | null;
  currentView: 'generator' | 'dashboard';
  scheduledPostsCount: number;
  onViewChange: (view: 'generator' | 'dashboard') => void;
  onSignOut?: () => void;
}

const GeneratorHeader: React.FC<GeneratorHeaderProps> = ({
  userId,
  currentView,
  scheduledPostsCount,
  onViewChange,
  onSignOut,
}) => {
  const navigate = useNavigate();

  const handleSignOut = async () => {
    if (onSignOut) {
      await onSignOut();
      navigate('/');
    }
  };

  return (
    <header className="p-4 md:p-6 bg-slate-900 border-b border-blue-900/50">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <div className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          Ghostwriter
        </div>
        <div className="flex items-center space-x-4">
          <p className="text-sm font-mono text-slate-500 hidden sm:block">
            User ID: <span className="text-blue-400 font-semibold">{userId || 'N/A'}</span>
          </p>
          <button
            onClick={() => onViewChange('generator')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition ${
              currentView === 'generator' 
                ? 'bg-blue-600 text-white' 
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Generator
          </button>
          <button
            onClick={() => onViewChange('dashboard')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition ${
              currentView === 'dashboard' 
                ? 'bg-emerald-600 text-white' 
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Scheduled Posts ({scheduledPostsCount})
          </button>
          {onSignOut && (
            <button
              onClick={handleSignOut}
              className="px-4 py-2 text-sm font-semibold rounded-lg bg-red-600 text-white hover:bg-red-700 transition"
            >
              Sign Out
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default GeneratorHeader;

