import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useUser } from '../hooks/useUser';
import UsageLimit from './UsageLimit';

const PrivateRoute = ({ children }) => {
  const { user, loading, canAccess, incrementUsage } = useUser();

  useEffect(() => {
    if (!user && canAccess()) {
      incrementUsage();
    }
  }, [user, canAccess, incrementUsage]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!canAccess()) {
    return <Navigate to="/" />;
  }

  return (
    <>
      {!user && <UsageLimit />}
      {children}
    </>
  );
};

export default PrivateRoute;