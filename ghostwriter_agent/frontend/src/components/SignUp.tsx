import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { BG_DARK, BG_MEDIUM } from '../constants/theme';

interface SignUpProps {
  onSignUp: (email: string, password: string, displayName?: string) => Promise<void>;
  onSwitchToLogin: () => void;
  error?: string | null;
}

const SignUp: React.FC<SignUpProps> = ({ onSignUp, onSwitchToLogin, error }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    
    if (!email || !password || !confirmPassword) {
      setLocalError('Please fill in all required fields');
      return;
    }

    if (password.length < 6) {
      setLocalError('Password must be at least 6 characters');
      return;
    }

    if (password !== confirmPassword) {
      setLocalError('Passwords do not match');
      return;
    }

    setIsLoading(true);
    try {
      await onSignUp(email, password, displayName || undefined);
      // Navigate to generator on success
      navigate('/generator');
    } catch (err: any) {
      setLocalError(err.message || 'Failed to create account. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const displayError = error || localError;

  return (
    <div className={`min-h-screen ${BG_DARK} flex items-center justify-center px-4`}>
      <div className={`w-full max-w-md p-8 rounded-2xl ${BG_MEDIUM} border border-emerald-600/50 shadow-2xl`}>
        <h2 className="text-3xl font-bold text-center mb-2 text-white">
          Create Account
        </h2>
        <p className="text-center text-slate-400 mb-8">
          Start your Ghostwriter journey
        </p>

        {displayError && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-600/50 rounded-lg text-red-300 text-sm">
            {displayError}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="displayName" className="block text-sm font-medium text-slate-300 mb-2">
              Display Name (Optional)
            </label>
            <input
              id="displayName"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className={`w-full px-4 py-3 ${BG_DARK} border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500`}
              placeholder="Your name"
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-2">
              Email Address *
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={`w-full px-4 py-3 ${BG_DARK} border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500`}
              placeholder="you@example.com"
              required
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-2">
              Password *
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={`w-full px-4 py-3 ${BG_DARK} border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500`}
              placeholder="••••••••"
              required
              disabled={isLoading}
            />
            <p className="mt-1 text-xs text-slate-500">Must be at least 6 characters</p>
          </div>

          <div>
            <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-300 mb-2">
              Confirm Password *
            </label>
            <input
              id="confirmPassword"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className={`w-full px-4 py-3 ${BG_DARK} border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500`}
              placeholder="••••••••"
              required
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-4 rounded-lg font-semibold transition ${
              isLoading
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
                : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-lg shadow-emerald-500/30'
            }`}
          >
            {isLoading ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>

        <div className="mt-6 text-center space-y-2">
          <p className="text-slate-400 text-sm">
            Already have an account?{' '}
            <Link
              to="/login"
              className="text-emerald-400 hover:text-emerald-300 font-semibold transition"
            >
              Sign in
            </Link>
          </p>
          <Link
            to="/"
            className="text-sm text-slate-400 hover:text-slate-300 transition block"
          >
            ← Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
};

export default SignUp;

