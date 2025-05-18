import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import theme from './theme';
import { UserProvider } from './context/UserContext';
import { SubscriptionProvider } from './context/SubscriptionContext';
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

function App() {
  return (
    <UserProvider>
      <SubscriptionProvider>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Router>
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
                  <Route path="/" element={<LandingPage />} />
                  <Route path="/login" element={<Login />} />
                  <Route path="/register" element={<Register />} />
                  <Route
                    path="/dashboard"
                    element={
                      <PrivateRoute>
                        <Dashboard />
                      </PrivateRoute>
                    }
                  />
                  <Route path="/upload" element={<DocumentUpload />} />
                  <Route path="/templates" element={<TemplateSelection />} />
                  <Route path="/preview/:documentId" element={<Preview />} />
                  <Route path="/about" element={<About />} />
                  <Route path="/contact" element={<Contact />} />
                  <Route path="/terms" element={<Terms />} />
                  <Route path="/template-management" element={<TemplateManagement />} />
                  <Route path="/upgrade" element={<Upgrade />} />
                  <Route
                    path="/profile"
                    element={
                      <PrivateRoute>
                        <Profile />
                      </PrivateRoute>
                    }
                  />
                  {/* Redirect /analysis to dashboard since the Analysis component doesn't exist */}
                  <Route path="/analysis" element={<Navigate to="/dashboard" replace />} />
                  {/* Catch all route for 404 */}
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
                <Footer />
              </Box>
            </Box>
          </Router>
        </ThemeProvider>
      </SubscriptionProvider>
    </UserProvider>
  );
}

export default App;
