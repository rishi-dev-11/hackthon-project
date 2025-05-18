import React from 'react';
import {
  Box,
  Paper,
  Alert,
  Button,
  Typography,
  LinearProgress,
} from '@mui/material';
import { Warning as WarningIcon } from '@mui/icons-material';
import { useSubscription } from '../context/SubscriptionContext';
import { useNavigate } from 'react-router-dom';

const UsageLimit = ({ type, showUpgradeButton = true }) => {
  const { usageStats, getUsageLimits, getRemainingQuota } = useSubscription();
  const navigate = useNavigate();
  
  const getUsagePercentage = () => {
    const remaining = getRemainingQuota(type);
    if (remaining === 'Unlimited') return 0;
    
    const limits = getUsageLimits('free');
    const total = limits[`${type}PerMonth`];
    const used = usageStats[type];
    return (used / total) * 100;
  };

  const percentage = getUsagePercentage();
  const isCritical = percentage >= 90;
  const isLimitReached = percentage >= 100;

  if (percentage < 80) return null;

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        mb: 2, 
        bgcolor: isLimitReached ? 'error.light' : 'warning.light',
        border: 1,
        borderColor: isLimitReached ? 'error.main' : 'warning.main'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <WarningIcon color={isLimitReached ? 'error' : 'warning'} />
        <Typography variant="body1" color={isLimitReached ? 'error' : 'warning'}>
          {isLimitReached 
            ? `You've reached your ${type} limit` 
            : `You're approaching your ${type} limit`}
        </Typography>
      </Box>
      
      <Alert severity="info" sx={{ mt: 2 }}>
        {isLimitReached 
          ? 'Upgrade your plan to continue using this feature'
          : `${Math.round(100 - percentage)} uses remaining this month`}
      </Alert>

      <LinearProgress
        variant="determinate"
        value={percentage}
        color={isCritical ? "error" : "warning"}
        sx={{
          height: 6,
          borderRadius: 3,
          mb: 1,
          bgcolor: 'rgba(0,0,0,0.05)'
        }}
      />
      
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
        {Math.round(percentage)}% used
      </Typography>

      {showUpgradeButton && (
        <Button
          variant="contained"
          size="small"
          onClick={() => navigate('/upgrade')}
          sx={{
            mt: 1,
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
          }}
        >
          Upgrade to Premium
        </Button>
      )}
    </Paper>
  );
};

export default UsageLimit;