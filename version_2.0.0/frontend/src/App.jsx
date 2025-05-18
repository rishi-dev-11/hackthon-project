import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Snackbar, Alert } from '@mui/material';
import theme from './theme';
import Login from './pages/Login';
import Register from './pages/Register';
import PrivateRoute from './components/PrivateRoute';

import HamburgerMenu from './components/HamburgerMenu';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import DocumentUpload from './pages/DocumentUpload';
import TemplateSelection from './pages/TemplateSelection';
import Preview from './pages/Preview';
import About from './pages/About';
import Contact from './pages/Contact';
import Terms from './pages/Terms';
import TemplateManagement from './pages/TemplateManagement';
import Profile from './pages/Profile';
import Upgrade from './pages/Upgrade';
import NotFound from './pages/NotFound';
import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import UserGuide from './components/UserGuide';

function App() {
  const [notification, setNotification] = useState(null);
  const location = useLocation();

  // Clear notifications when route changes
  useEffect(() => {
    setNotification(null);
  }, [location.pathname]);

  // Show a notification for new users
  useEffect(() => {
    const isNewUser = !localStorage.getItem('hasVisitedBefore');
    if (isNewUser && location.pathname === '/') {
      setNotification({
        message: 'Welcome to DocuMorph AI! Explore our document transformation features.',
        severity: 'info'
      });
      localStorage.setItem('hasVisitedBefore', 'true');
    }
  }, [location.pathname]);

  const handleCloseNotification = () => {
    setNotification(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          display: 'flex',
          minHeight: '100vh',
          width: '100%',
          overflow: 'hidden'
        }}
      >
        <HamburgerMenu />
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            width: '100%',
            minHeight: '100vh',
            pt: '64px', // Height of AppBar
            overflow: 'auto'
          }}
        >
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/terms" element={<Terms />} />
            
            {/* Auth-protected routes */}
            <Route
              path="/dashboard"
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              }
            />
            <Route 
              path="/profile"
              element={
                <PrivateRoute>
                  <Profile />
                </PrivateRoute>
              }
            />
            <Route 
              path="/template-management"
              element={
                <PrivateRoute>
                  <TemplateManagement />
                </PrivateRoute>
              }
            />
            
            {/* Semi-protected routes (can view with limited functionality) */}
            <Route path="/upload" element={<DocumentUpload />} />
            <Route path="/templates" element={<TemplateSelection />} />
            <Route path="/preview/:documentId" element={<Preview />} />
            <Route path="/upgrade" element={<Upgrade />} />
            
            {/* Placeholder routes with proper feedback */}
            <Route 
              path="/style" 
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/analysis" 
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } 
            />
            <Route 
              path="/tables" 
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } 
            />
            
            {/* 404 NotFound route */}
            <Route path="/not-found" element={<NotFound />} />
            
            {/* Catch-all route to redirect to NotFound page */}
            <Route path="*" element={<Navigate to="/not-found" replace />} />
          </Routes>
          
          <Footer />
          
          {/* Global notification system */}
          <Snackbar 
            open={!!notification} 
            autoHideDuration={6000} 
            onClose={handleCloseNotification}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            {notification && (
              <Alert 
                onClose={handleCloseNotification} 
                severity={notification.severity || 'info'}
                sx={{ width: '100%' }}
              >
                {notification.message}
              </Alert>
            )}
          </Snackbar>
          
          {/* Context-aware user guide */}
          <UserGuide />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
