import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

export const UserContext = createContext(null);

export const useUser = () => useContext(UserContext);

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [subscription, setSubscription] = useState('free');
  const [usageCount, setUsageCount] = useState(() => {
    const savedCount = localStorage.getItem('usageCount');
    return savedCount ? parseInt(savedCount) : 0;
  });

  const MAX_FREE_USES = 2;
  
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const incrementUsage = () => {
    setUsageCount(prevCount => {
      const newCount = prevCount + 1;
      localStorage.setItem('usageCount', newCount.toString());
      return newCount;
    });
  };

  const canAccess = () => {
    return user !== null || usageCount < MAX_FREE_USES;
  };

  const getUserSubscription = () => {
    if (!user) return 'free';
    return subscription;
  };

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      if (token) {
        const response = await axios.get('http://localhost:8000/api/users/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setUser(response.data);
        // Set subscription based on user data
        setSubscription(response.data.subscription || 'free');
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      localStorage.removeItem('token');
      setSubscription('free');
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const response = await axios.post('http://localhost:8000/api/users/login', {
        email,
        password
      });
      const { token, user } = response.data;
      localStorage.setItem('token', token);
      setUser(user);
      // Reset usage count after successful login
      setUsageCount(0);
      localStorage.removeItem('usageCount');
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.message || 'Login failed'
      };
    }
  };

  const register = async (userData) => {
    try {
      const response = await axios.post('http://localhost:8000/api/users/register', userData);
      const { token, user } = response.data;
      localStorage.setItem('token', token);
      setUser(user);
      // Reset usage count after successful registration
      setUsageCount(0);
      localStorage.removeItem('usageCount');
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.message || 'Registration failed'
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const updateProfile = async (updates) => {
    try {
      const updatedUser = { ...user, ...updates };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
      return { success: true };
    } catch (error) {
      console.error('Profile update failed:', error);
      return { success: false, error: error.message };
    }
  };

  return (
    <UserContext.Provider
      value={{
        user,
        setUser,
        loading,
        login,
        logout,
        register,
        updateProfile,
        checkAuthStatus,
        canAccess,
        incrementUsage,
        usageCount,
        subscription,
        getUserSubscription
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export default UserContext;