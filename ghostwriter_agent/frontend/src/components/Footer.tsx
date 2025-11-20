import React from 'react';

interface FooterProps {
  userId?: string | null;
}

const Footer: React.FC<FooterProps> = ({ userId }) => {
  return (
    <footer className="p-8 bg-slate-950 text-center text-slate-500 text-sm">
      <div className="max-w-7xl mx-auto">
        <p>
          &copy; 2025 Ghostwriter. All Rights Reserved. | Dedicated to Editorial Excellence.
          {userId && ` | Auth UID: ${userId}`}
        </p>
      </div>
    </footer>
  );
};

export default Footer;

