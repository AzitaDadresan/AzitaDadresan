import { Navigate } from 'react-router-dom';
import { useMockFirebase } from '../hooks/useMockFirebase';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { userId, isAuthReady } = useMockFirebase();

  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <p className="text-xl text-blue-300">Initializing...</p>
      </div>
    );
  }

  if (!userId) {
    // Redirect to landing page if not authenticated
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;

