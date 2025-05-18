import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  Button,
  Divider,
  Container,
  Typography,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  CloudUpload,
  Description,
  Info,
  Gavel,
  Person,
  Home,
  Settings,
  ExitToApp,
  AutoStories,
  SmartToy,
  Login as LoginIcon,
  PersonAdd as RegisterIcon,
  Stars as StarsIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../hooks/useUser';
import { useSubscription } from '../hooks/useSubscription';
import UserProfile from './UserProfile';

const drawerWidth = 280;

const HamburgerMenu = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { user } = useUser();
  const { userTier } = useSubscription();
  const navigate = useNavigate();
  const isPremium = userTier === 'premium';

  const menuItems = [
    { 
      text: 'Home', 
      icon: <Home />, 
      path: '/',
      color: '#2196F3' // Blue
    },
    { 
      text: 'Dashboard', 
      icon: <Dashboard />, 
      path: '/dashboard',
      color: '#4CAF50' // Green
    },
    { 
      text: 'Upload Document', 
      icon: <CloudUpload />, 
      path: '/upload',
      color: '#FF9800' // Orange
    },
    { 
      text: 'Templates', 
      icon: <AutoStories />, 
      path: '/templates',
      color: '#9C27B0' // Purple
    },
    { 
      text: 'AI Analysis', 
      icon: <SmartToy />, 
      path: '/analysis',
      color: '#E91E63' // Pink
    },
    { 
      text: 'About', 
      icon: <Info />, 
      path: '/about',
      color: '#00BCD4' // Cyan
    },
    { 
      text: 'Terms', 
      icon: <Gavel />, 
      path: '/terms',
      color: '#F44336' // Red
    },
    { 
      text: 'Upgrade to Premium', 
      icon: <StarsIcon />, 
      path: '/upgrade',
      color: '#FF9800', // Orange
      showWhen: !isPremium
    },
  ];

  const handleNavigation = (path) => {
    navigate(path);
    setDrawerOpen(false);
  };

  return (
    <>
      <AppBar 
        position="fixed" 
        sx={{ 
          bgcolor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 2px 15px rgba(0,0,0,0.1)',
          width: '100%',
          left: 0,
          right: 0,
          zIndex: 1200
        }}
      >
        <Container maxWidth={false}>
          <Toolbar sx={{ 
            justifyContent: 'space-between',
            width: '100%',
            px: { xs: 1, sm: 2, md: 3 },
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <IconButton
                edge="start"
                color="inherit"
                onClick={() => setDrawerOpen(!drawerOpen)}
                sx={{
                  color: 'primary.main',
                  '&:hover': {
                    transform: 'rotate(90deg) scale(1.1)',
                    transition: 'all 0.3s ease-in-out',
                    bgcolor: 'rgba(33, 150, 243, 0.1)',
                  },
                }}
              >
                <MenuIcon />
              </IconButton>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 'bold',
                  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  backgroundClip: 'text',
                  textFillColor: 'transparent',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  display: { xs: 'none', sm: 'block' },
                  fontSize: { sm: '1.25rem', md: '1.5rem' },
                }}
              >
                DocuMorph AI
              </Typography>
            </Box>
            
            {user && (
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 2,
              }}>
                <UserProfile />
              </Box>
            )}
          </Toolbar>
        </Container>
      </AppBar>

      <Drawer
        variant="temporary"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            borderRight: '1px solid',
            borderColor: 'divider',
            boxShadow: '4px 0 10px rgba(0,0,0,0.1)',
            bgcolor: 'background.paper',
          }
        }}
      >
        <Box sx={{ 
          overflow: 'auto',
          height: '100%',
          pt: 8 // Space for AppBar
        }}>
          <List>
            {menuItems
              .filter(item => !item.showWhen || item.showWhen)
              .map((item) => (
                <ListItem
                  disableRipple
                  key={item.text}
                  onClick={() => handleNavigation(item.path)}
                  sx={{
                    borderRadius: '12px',
                    mb: 1,
                    transition: 'all 0.3s ease',
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: `${item.color}15`,
                      color: item.color,
                      transform: 'translateX(8px) scale(1.02)',
                      boxShadow: `0 4px 12px ${item.color}30`,
                      '& .MuiListItemIcon-root': {
                        transform: 'scale(1.2) rotate(5deg)',
                        color: item.color,
                      },
                    },
                  }}
                >
                  <ListItemIcon sx={{ 
                    color: 'inherit',
                    transition: 'all 0.3s ease',
                  }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.text}
                    primaryTypographyProps={{
                      sx: {
                        fontWeight: 500,
                        transition: 'all 0.3s ease',
                      }
                    }}
                  />
                </ListItem>
              ))}
          </List>

          <Divider sx={{ my: 2 }} />

          {!user ? (
            <>
              <ListItem
                disableRipple
                onClick={() => handleNavigation('/login')}
                sx={{
                  borderRadius: '12px',
                  mb: 1,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: '#2196F315',
                    color: '#2196F3',
                    transform: 'translateX(8px) scale(1.02)',
                    boxShadow: '0 4px 12px rgba(33, 150, 243, 0.2)',
                    '& .MuiListItemIcon-root': {
                      transform: 'scale(1.2) rotate(5deg)',
                      color: '#2196F3',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: 'inherit',
                  transition: 'all 0.3s ease',
                }}>
                  <LoginIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Login"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 500,
                      transition: 'all 0.3s ease',
                    }
                  }}
                />
              </ListItem>
              <ListItem
                disableRipple
                onClick={() => handleNavigation('/register')}
                sx={{
                  borderRadius: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: '#4CAF5015',
                    color: '#4CAF50',
                    transform: 'translateX(8px) scale(1.02)',
                    boxShadow: '0 4px 12px rgba(76, 175, 80, 0.2)',
                    '& .MuiListItemIcon-root': {
                      transform: 'scale(1.2) rotate(5deg)',
                      color: '#4CAF50',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: 'inherit',
                  transition: 'all 0.3s ease',
                }}>
                  <RegisterIcon />
                </ListItemIcon>
                <ListItemText 
                  primary="Register"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 500,
                      transition: 'all 0.3s ease',
                    }
                  }}
                />
              </ListItem>
            </>
          ) : (
            <>
              <ListItem
                disableRipple
                onClick={() => handleNavigation('/profile')}
                sx={{
                  borderRadius: '12px',
                  mb: 1,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: '#2196F315',
                    color: '#2196F3',
                    transform: 'translateX(8px) scale(1.02)',
                    boxShadow: '0 4px 12px rgba(33, 150, 243, 0.2)',
                    '& .MuiListItemIcon-root': {
                      transform: 'scale(1.2) rotate(5deg)',
                      color: '#2196F3',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: 'inherit',
                  transition: 'all 0.3s ease',
                }}>
                  <Person />
                </ListItemIcon>
                <ListItemText 
                  primary="Profile"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 500,
                      transition: 'all 0.3s ease',
                    }
                  }}
                />
              </ListItem>
              <ListItem
                disableRipple
                onClick={() => handleNavigation('/settings')}
                sx={{
                  borderRadius: '12px',
                  mb: 1,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: '#FF980015',
                    color: '#FF9800',
                    transform: 'translateX(8px) scale(1.02)',
                    boxShadow: '0 4px 12px rgba(255, 152, 0, 0.2)',
                    '& .MuiListItemIcon-root': {
                      transform: 'scale(1.2) rotate(5deg)',
                      color: '#FF9800',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: 'inherit',
                  transition: 'all 0.3s ease',
                }}>
                  <Settings />
                </ListItemIcon>
                <ListItemText 
                  primary="Settings"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 500,
                      transition: 'all 0.3s ease',
                    }
                  }}
                />
              </ListItem>
              <ListItem
                disableRipple
                onClick={() => handleNavigation('/logout')}
                sx={{
                  borderRadius: '12px',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    bgcolor: '#F4433615',
                    color: '#F44336',
                    transform: 'translateX(8px) scale(1.02)',
                    boxShadow: '0 4px 12px rgba(244, 67, 54, 0.2)',
                    '& .MuiListItemIcon-root': {
                      transform: 'scale(1.2) rotate(5deg)',
                      color: '#F44336',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: 'inherit',
                  transition: 'all 0.3s ease',
                }}>
                  <ExitToApp />
                </ListItemIcon>
                <ListItemText 
                  primary="Logout"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 500,
                      transition: 'all 0.3s ease',
                    }
                  }}
                />
              </ListItem>
            </>
          )}
        </Box>
      </Drawer>
    </>
  );
};

export default HamburgerMenu;