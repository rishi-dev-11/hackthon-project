import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Chip
} from '@mui/material';
import { useUser } from '../hooks/useUser';
import { useNavigate } from 'react-router-dom';
import { Stars as StarsIcon, Lock as LockIcon } from '@mui/icons-material';

const FeatureLimit = ({ featureKey, message }) => {
  const { getFeatureAccess, getUserSubscription } = useUser();
  const navigate = useNavigate();
  const feature = getFeatureAccess(featureKey);
  const currentSubscription = getUserSubscription();

  if (feature.allowed && currentSubscription === 'premium') {
    return null; // Don't show anything for premium users with access
  }

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 3,
        mt: 2,
        borderRadius: 2,
        bgcolor: 'rgba(33, 150, 243, 0.05)',
        border: '1px solid',
        borderColor: 'primary.main',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <LockIcon color="primary" />
        <Typography variant="h6">
          {feature.allowed ? 'Feature Limitation' : 'Premium Feature'}
        </Typography>
        <Chip 
          label={currentSubscription === 'premium' ? 'Premium' : 'Free'} 
          color={currentSubscription === 'premium' ? 'primary' : 'default'}
          size="small"
        />
      </Box>

      <Typography paragraph>
        {message || feature.description}
      </Typography>

      <Typography color="text.secondary" paragraph>
        Current limit: {feature.limit}
      </Typography>

      {currentSubscription !== 'premium' && (
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={() => navigate('/upgrade')}
            startIcon={<StarsIcon />}
            sx={{
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
            }}
          >
            Upgrade to Premium
          </Button>
          <Button
            variant="outlined"
            color="primary"
            onClick={() => navigate('/features')}
          >
            Compare Plans
          </Button>
        </Box>
      )}
    </Paper>
  );
};

export default FeatureLimit;