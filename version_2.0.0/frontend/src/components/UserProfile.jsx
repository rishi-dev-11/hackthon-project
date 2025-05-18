import React, { useState } from 'react';
import {
  Avatar,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Typography,
  Divider,
  ListItemIcon,
} from '@mui/material';
import {
  AccountCircle,
  Person,
  Settings,
  ExitToApp,
  Stars as StarsIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../hooks/useUser';
import { useSubscription } from '../hooks/useSubscription';

const UserProfile = () => {
  const [anchorEl, setAnchorEl] = useState(null);
  const { user, logout } = useUser();
  const { userTier } = useSubscription();
  const navigate = useNavigate();

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleClose();
    logout();
    navigate('/');
  };

  const handleProfile = () => {
    handleClose();
    navigate('/profile');
  };

  return (
    <Box>
      <IconButton
        size="large"
        onClick={handleMenu}
        color="inherit"
        sx={{
          p: 1,
          '&:hover': {
            bgcolor: 'rgba(0,0,0,0.04)',
          },
        }}
      >
        {user?.avatar ? (
          <Avatar 
            src={user.avatar}
            alt={user?.name || "User"}
            sx={{ width: 40, height: 40 }}
          />
        ) : (
          <AccountCircle 
            sx={{ 
              width: 40, 
              height: 40, 
              color: 'primary.main' 
            }} 
          />
        )}
      </IconButton>
      <Menu
        anchorEl={anchorEl}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        keepMounted
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        PaperProps={{
          sx: { 
            mt: 1.5, 
            width: 220,
            borderRadius: 2,
            boxShadow: 3
          }
        }}
      >
        <Box sx={{ py: 1, px: 2, mb: 1 }}>
          <Typography variant="subtitle1" noWrap>
            {user?.name || "User"}
          </Typography>
          <Typography variant="body2" color="text.secondary" noWrap>
            {user?.email || "user@example.com"}
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <StarsIcon
              fontSize="small"
              sx={{
                color: userTier === 'premium' ? 'gold' : 'text.disabled',
              }}
            />
            <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
              {userTier} tier
            </Typography>
          </Box>
        </Box>
        
        <Divider />
        
        <MenuItem onClick={handleProfile} sx={{ py: 1.5 }}>
          <ListItemIcon>
            <Person fontSize="small" />
          </ListItemIcon>
          Profile
        </MenuItem>
        
        <MenuItem onClick={() => { handleClose(); navigate('/settings'); }} sx={{ py: 1.5 }}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        
        {userTier !== 'premium' && (
          <MenuItem onClick={() => { handleClose(); navigate('/upgrade'); }} sx={{ py: 1.5 }}>
            <ListItemIcon>
              <StarsIcon fontSize="small" sx={{ color: 'gold' }} />
            </ListItemIcon>
            Upgrade to Premium
          </MenuItem>
        )}
        
        <Divider />
        
        <MenuItem onClick={handleLogout} sx={{ py: 1.5 }}>
          <ListItemIcon>
            <ExitToApp fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default UserProfile;