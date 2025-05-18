import React, { useState } from 'react';
import {
  Box,
  Avatar,
  Menu,
  MenuItem,
  IconButton,
  Typography,
  Divider,
  ListItemIcon,
  Chip,
  Button,
} from '@mui/material';
import {
  AccountCircle,
  Logout,
  Settings,
  Person,
  Stars as StarsIcon,
} from '@mui/icons-material';
import { useUser } from '../hooks/useUser';
import { useNavigate } from 'react-router-dom';

const UserProfile = () => {
  const [anchorEl, setAnchorEl] = useState(null);
  // const { user, logout, getUserSubscription } = useUser();
  const navigate = useNavigate();
  // const subscriptionType = getUserSubscription();
const { user, logout } = useUser();
const subscriptionType = user?.subscription || 'free';

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleClose();
    navigate('/');
  };

  const handleProfile = () => {
    handleClose();
    navigate('/profile');
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {subscriptionType === 'premium' && (
          <Chip
            icon={<StarsIcon />}
            label="Premium"
            size="small"
            color="primary"
            sx={{
              mr: 1,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              color: 'white',
            }}
          />
        )}
        <IconButton
          size="large"
          onClick={handleMenu}
          color="inherit"
          sx={{ ml: 1 }}
        >
          {user?.avatar ? (
            <Avatar src={user.avatar} alt={user.name} />
          ) : (
            <AccountCircle />
          )}
        </IconButton>
      </Box>
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
            width: 250,
            '& .MuiAvatar-root': {
              width: 32,
              height: 32,
              ml: -0.5,
              mr: 1,
            },
          },
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
            {user?.name || 'User'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {user?.email || 'user@example.com'}
          </Typography>
          {subscriptionType === 'free' && (
            <Button
              variant="contained"
              size="small"
              startIcon={<StarsIcon />}
              onClick={() => {
                handleClose();
                navigate('/upgrade');
              }}
              sx={{
                mt: 1,
                width: '100%',
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
              }}
            >
              Upgrade to Premium
            </Button>
          )}
        </Box>
        <Divider />
        <MenuItem onClick={handleProfile}>
          <ListItemIcon>
            <Person fontSize="small" />
          </ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem onClick={() => {
          handleClose();
          navigate('/settings');
        }}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default UserProfile;