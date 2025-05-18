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
  Divider,
  Container,
  Typography,
  Avatar,
  Tooltip,
  Button,
  Chip,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  CloudUpload,
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
  Description,
  History,
  Psychology,
  Groups,
  School,
  MenuBook,
  FormatAlignLeft,
  ViewComfy,
  TableChart,
  ContactSupport,
  ArrowForwardIos
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useUser } from '../hooks/useUser';
import { useSubscription } from '../hooks/useSubscription';
import UserProfile from './UserProfile';

const drawerWidth = 280;

const HamburgerMenu = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { user, isAuthenticated, logout } = useUser();
  const { userTier } = useSubscription();
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // Group menu items by category for better organization
  const menuCategories = [
    {
      title: 'Main',
      items: [
        { 
          text: 'Home', 
          icon: <Home />, 
          path: '/',
          color: '#0E6BA8', // Streamlit blue
          description: 'Go to the home page'
        },
        { 
          text: 'Dashboard', 
          icon: <Dashboard />, 
          path: '/dashboard',
          color: '#4CAF50', // Green
          requiresAuth: true,
          description: 'Access your dashboard'
        },
      ]
    },
    {
      title: 'Documents',
      items: [
        { 
          text: 'Upload Document', 
          icon: <CloudUpload />, 
          path: '/upload',
          color: '#FF9800', // Orange
          description: 'Upload a new document'
        },
        { 
          text: 'Document History', 
          icon: <History />, 
          path: '/history',
          color: '#9C27B0', // Purple
          requiresAuth: true,
          description: 'View your document history'
        },
        { 
          text: 'Templates', 
          icon: <AutoStories />, 
          path: '/templates',
          color: '#3F51B5', // Indigo
          description: 'Explore document templates'
        },
      ]
    },
    {
      title: 'AI Features',
      items: [
        { 
          text: 'Document Analysis', 
          icon: <SmartToy />, 
          path: '/analysis',
          color: '#E91E63', // Pink
          requiresPremium: userTier === 'free',
          description: 'Analyze a document'
        },
        { 
          text: 'Style Enhancement', 
          icon: <FormatAlignLeft />, 
          path: '/style',
          color: '#4A9BE5', // Light blue
          description: 'Enhance the style of a document'
        },
        { 
          text: 'Table Detection', 
          icon: <TableChart />, 
          path: '/tables',
          color: '#FF5722', // Deep orange
          requiresPremium: userTier === 'free',
          description: 'Detect tables in a document'
        },
      ]
    },
    {
      title: 'Account',
      items: [
        { 
          text: 'Profile', 
          icon: <Person />, 
          path: '/profile',
          color: '#00BCD4', // Cyan
          requiresAuth: true,
          description: 'View your profile'
        },
        { 
          text: 'Upgrade to Premium', 
          icon: <StarsIcon />, 
          path: '/upgrade',
          color: '#FF4B4B', // Streamlit accent red
          showWhen: userTier === 'free',
          description: 'Upgrade to a premium account'
        },
      ]
    },
    {
      title: 'Information',
      items: [
        { 
          text: 'About', 
          icon: <Info />, 
          path: '/about',
          color: '#607D8B', // Blue gray
          description: 'Learn about us'
        },
        { 
          text: 'Terms', 
          icon: <Gavel />, 
          path: '/terms',
          color: '#795548', // Brown
          description: 'Read our terms of service'
        },
        { 
          text: 'Contact Us', 
          icon: <ContactSupport />, 
          path: '/contact',
          color: '#009688', // Teal 
          description: 'Get in touch with us'
        },
      ]
    }
  ];

  const isActive = (path) => {
    return location.pathname === path;
  };

  const handleNavigation = (path) => {
    navigate(path);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  return (
    <>
      <AppBar 
        position="fixed" 
        sx={{ 
          bgcolor: 'background.paper',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
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
            minHeight: { xs: 56, sm: 64 }
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <IconButton
                edge="start"
                color="inherit"
                onClick={() => setDrawerOpen(!drawerOpen)}
                sx={{
                  color: 'primary.main',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'rotate(180deg)',
                    bgcolor: 'rgba(14, 107, 168, 0.08)',
                  },
                }}
              >
                <MenuIcon />
              </IconButton>
              <Box 
                onClick={() => navigate('/')}
                sx={{ 
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 700,
                    color: 'primary.main',
                    display: { xs: 'none', sm: 'block' },
                    fontSize: { sm: '1.25rem', md: '1.5rem' },
                  }}
                >
                  DocuMorph AI
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 2,
            }}>
              {isAuthenticated ? (
                <UserProfile />
              ) : (
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    color="primary"
                    onClick={() => handleNavigation('/login')}
                    sx={{ display: { xs: 'none', sm: 'flex' } }}
                  >
                    Sign In
                  </Button>
                  <Button
                    variant="contained"
                    size="small"
                    color="primary"
                    onClick={() => handleNavigation('/register')}
                  >
                    Sign Up
                  </Button>
                </Box>
              )}
            </Box>
          </Toolbar>
        </Container>
      </AppBar>

      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={isMobile ? drawerOpen : true}
        onClose={() => setDrawerOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            borderRight: '1px solid',
            borderColor: 'divider',
            boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
            bgcolor: 'background.paper',
            pt: 8, // Space for AppBar
            height: '100%',
            zIndex: 1100
          },
          display: { xs: isMobile ? 'block' : 'none', md: 'block' }
        }}
      >
        <Box sx={{ 
          overflow: 'auto',
          height: '100%',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {isAuthenticated && (
            <Box sx={{ px: 3, py: 2, textAlign: 'center' }}>
              <Avatar 
                src={user?.profilePic || ""}
                alt={user?.name || "User"} 
                sx={{ 
                  width: 80, 
                  height: 80, 
                  mx: 'auto',
                  mb: 1,
                  boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                }}
              />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {user?.name || "User"}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {user?.email || "user@example.com"}
              </Typography>
              <Chip 
                label={userTier === 'premium' ? 'Premium' : 'Free'} 
                color={userTier === 'premium' ? 'secondary' : 'primary'}
                size="small"
                sx={{ mt: 0.5 }}
              />
            </Box>
          )}
          
          {menuCategories.map((category) => (
            <React.Fragment key={category.title}>
              <Divider sx={{ my: 1.5 }} />
              
              <Typography 
                variant="overline" 
                sx={{ 
                  pl: 3, 
                  color: 'text.secondary',
                  fontWeight: 600,
                  display: 'block',
                  fontSize: '0.75rem',
                  letterSpacing: '0.08em'
                }}
              >
                {category.title}
              </Typography>
              
              <List>
                {category.items
                  .filter(item => !item.requiresAuth || isAuthenticated)
                  .filter(item => item.showWhen === undefined || item.showWhen)
                  .map((item) => (
                    <ListItem
                      component={Button}
                      disableRipple
                      key={item.text}
                      onClick={() => handleNavigation(item.path)}
                      disabled={item.requiresPremium}
                      title={item.requiresPremium ? "Upgrade to Premium to access this feature" : `Go to ${item.text}`}
                      sx={{
                        borderRadius: 0,
                        px: 2,
                        py: 0.75,
                        borderLeft: '4px solid',
                        borderLeftColor: isActive(item.path) ? item.color : 'transparent',
                        position: 'relative',
                        bgcolor: isActive(item.path) ? `${item.color}10` : 'transparent',
                        '&:hover': {
                          bgcolor: `${item.color}10`,
                          '& .MuiListItemIcon-root': {
                            color: item.color,
                          },
                        },
                      }}
                    >
                      <ListItemIcon sx={{ 
                        color: isActive(item.path) ? item.color : 'inherit',
                        minWidth: 40,
                        opacity: item.requiresPremium ? 0.6 : 1
                      }}>
                        {item.icon}
                      </ListItemIcon>
                      <ListItemText 
                        primary={item.text}
                        secondary={item.description || null}
                        primaryTypographyProps={{
                          sx: {
                            fontWeight: isActive(item.path) ? 600 : 400,
                            color: isActive(item.path) ? item.color : 'text.primary',
                            fontSize: '0.95rem',
                            opacity: item.requiresPremium ? 0.6 : 1
                          }
                        }}
                        secondaryTypographyProps={{
                          sx: {
                            opacity: item.requiresPremium ? 0.6 : 1,
                            fontSize: '0.75rem'
                          }
                        }}
                      />
                      {item.requiresPremium && (
                        <Chip 
                          size="small" 
                          label="PRO" 
                          color="secondary" 
                          title="Premium feature"
                          sx={{ 
                            height: 20, 
                            fontSize: '0.65rem',
                            ml: 1
                          }} 
                        />
                      )}
                      {isActive(item.path) && (
                        <ArrowForwardIos sx={{ 
                          fontSize: 12, 
                          position: 'absolute',
                          right: 16,
                          color: item.color
                        }} />
                      )}
                    </ListItem>
                  ))}
              </List>
            </React.Fragment>
          ))}

          {isAuthenticated && (
            <>
              <Divider sx={{ mt: 'auto', mb: 1 }} />
              <ListItem
                component={Button}
                disableRipple
                onClick={logout}
                sx={{
                  borderRadius: 0,
                  px: 2,
                  py: 0.75,
                  '&:hover': {
                    bgcolor: 'rgba(244, 67, 54, 0.1)',
                    '& .MuiListItemIcon-root': {
                      color: 'error.main',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  <ExitToApp color="error" />
                </ListItemIcon>
                <ListItemText 
                  primary="Logout"
                  primaryTypographyProps={{
                    sx: {
                      fontWeight: 400,
                      color: 'error.main',
                      fontSize: '0.95rem'
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