import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Menu, MenuItem, Chip } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { useSubscription } from '../context/SubscriptionContext';

function Navbar() {
  const { userTier, isDevMode, switchPlan } = useSubscription();
  const [anchorEl, setAnchorEl] = useState(null);
  
  const handleDevMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleDevMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleSwitchPlan = (plan) => {
    switchPlan(plan);
    handleDevMenuClose();
  };
  
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          DocuMorph AI
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button color="inherit" component={RouterLink} to="/">
            Dashboard
          </Button>
          <Button color="inherit" component={RouterLink} to="/upload">
            Upload
          </Button>
          <Button color="inherit" component={RouterLink} to="/templates">
            Templates
          </Button>
          
          {isDevMode && (
            <>
              <Chip 
                label={`${userTier.charAt(0).toUpperCase() + userTier.slice(1)} Plan`}
                color={userTier === 'premium' ? 'secondary' : 'default'}
                onClick={handleDevMenuOpen}
                sx={{ cursor: 'pointer' }}
              />
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleDevMenuClose}
              >
                <MenuItem 
                  onClick={() => handleSwitchPlan('free')}
                  disabled={userTier === 'free'}
                >
                  Switch to Free Plan
                </MenuItem>
                <MenuItem 
                  onClick={() => handleSwitchPlan('premium')}
                  disabled={userTier === 'premium'}
                >
                  Switch to Premium Plan
                </MenuItem>
              </Menu>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar;
