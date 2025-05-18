import React from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Grid,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Info as InfoIcon,
  FileCopy as DocumentIcon,
  Storage as StorageIcon,
  Description as TemplateIcon,
} from '@mui/icons-material';
import { useSubscription } from '../context/SubscriptionContext';
import { useUser } from '../hooks/useUser';

const UsageMeter = ({ used, total, label, icon: Icon }) => {
  const percentage = total === 'Unlimited' ? 0 : Math.min(100, (used / total) * 100);
  const remaining = total === 'Unlimited' ? 'Unlimited' : Math.max(0, total - used);

  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        {Icon && <Icon sx={{ mr: 1, color: 'primary.main' }} />}
        <Typography variant="subtitle2">{label}</Typography>
        <Tooltip title="Monthly usage limit">
          <IconButton size="small" sx={{ ml: 'auto' }}>
            <InfoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        sx={{
          height: 8,
          borderRadius: 4,
          mb: 1,
          bgcolor: 'rgba(0,0,0,0.05)',
        }}
      />
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="caption" color="text.secondary">
          {used} used
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {remaining} remaining
        </Typography>
      </Box>
    </Box>
  );
};

const UsageStatistics = () => {
  const { usageStats, getRemainingQuota, getUsageLimits } = useSubscription();
  const { getUserSubscription } = useUser();
  const currentSubscription = getUserSubscription();
  const limits = getUsageLimits(currentSubscription);

  const features = ['document-processing', 'custom-templates'];

  return (
    <Paper elevation={0} sx={{ p: 3, borderRadius: 2 }}>
      <Typography variant="h6" gutterBottom>
        Usage Statistics
      </Typography>
      
      <Grid container spacing={2}>
        {features.map(feature => {
          const access = checkFeatureAccess(feature);
          if (!access.allowed) return null;
          
          const usageCount = usage[feature] || 0;
          const limit = access.limit;
          const percentage = limit === -1 ? 0 : (usageCount / limit) * 100;

          return (
            <Grid item xs={12} md={6} key={feature}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" sx={{ textTransform: 'capitalize' }}>
                  {feature.replace('-', ' ')}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Box sx={{ flexGrow: 1, mr: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={percentage}
                      color={percentage > 80 ? 'error' : 'primary'}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {limit === -1 ? 
                      `${usageCount} (Unlimited)` : 
                      `${usageCount}/${limit}`
                    }
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          );
        })}
      </Grid>

      {currentSubscription === 'trial' && (
        <Box sx={{ 
          mt: 2, 
          p: 2, 
          borderRadius: 1,
          bgcolor: 'primary.light',
          color: 'primary.contrastText'
        }}>
          <Typography variant="subtitle2" gutterBottom>
            Trial Period Active
          </Typography>
          <Typography variant="body2">
            {getRemainingQuota('trial')} days remaining in your trial period
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default UsageStatistics;