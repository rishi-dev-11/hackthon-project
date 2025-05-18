import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Chip,
  Tooltip
} from '@mui/material';
import { useUser } from '../hooks/useUser';
import { useSubscription } from '../context/SubscriptionContext';
import { useNavigate } from 'react-router-dom';
import { Stars as StarsIcon, Lock as LockIcon, Info as InfoIcon } from '@mui/icons-material';

const FeatureGuard = ({ feature, children, fallback }) => {
  const { getUserSubscription } = useUser();
  const { checkFeatureAccess, getUsageLimits } = useSubscription();
  const navigate = useNavigate();
  
  const currentSubscription = getUserSubscription();
  const limits = getUsageLimits(currentSubscription);
  const featureDetails = limits.features[feature];
  const access = checkFeatureAccess(feature);

  if (access.hasAccess) {
    return children;
  }

  if (fallback) {
    return fallback;
  }

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 3,
        borderRadius: 2,
        border: '1px solid #ddd',
        textAlign: 'center',
        bgcolor: 'background.paper',
        borderColor: 'primary.main',
      }}
    >
      <Box sx={{ mb: 2 }}>
        <LockIcon sx={{ fontSize: 40, color: 'grey.500' }} />
      </Box>
      <Typography variant="h6" gutterBottom>
        Premium Feature
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        This feature is only available on {access.requiredPlan} plan or higher
      </Typography>
      <Button
        variant="contained"
        color="primary"
        href="/upgrade"
      >
        Upgrade Now
      </Button>
    </Paper>
  );
};

export default FeatureGuard;