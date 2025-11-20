import { Navigate } from 'react-router-dom';
import { useMockFirebase } from '../hooks/useMockFirebase';

interface PublicRouteProps {
  children: React.ReactNode;
}

const PublicRoute: React.FC<PublicRouteProps> = ({ children }) => {
  const { userId, isAuthReady } = useMockFirebase();

  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <p className="text-xl text-blue-300">Initializing...</p>
      </div>
    );
  }

  // If user is logged in, redirect to generator instead of showing auth pages
  if (userId) {
    return <Navigate to="/generator" replace />;
  }

  return <>{children}</>;
};

export default PublicRoute;

