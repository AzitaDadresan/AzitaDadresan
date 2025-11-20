import React from 'react';
import { Link } from 'react-router-dom';

const Header: React.FC = () => {
  return (
    <header className="p-4 md:p-6 bg-slate-900 border-b border-blue-900/50">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <Link to="/" className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          Ghostwriter
        </Link>
        <div className="flex items-center space-x-3">
          <Link
            to="/login"
            className="text-sm font-semibold px-4 py-2 rounded-lg border border-blue-500 text-blue-300 hover:bg-blue-900/50 transition"
          >
            Sign In
          </Link>
          <Link
            to="/signup"
            className="text-sm font-semibold px-6 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
          >
            Sign Up
          </Link>
        </div>
      </div>
    </header>
  );
};

export default Header;

