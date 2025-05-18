import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  Tooltip
} from '@mui/material';
import { useUser } from '../hooks/useUser';
import { useNavigate } from 'react-router-dom';
import {
  Check as CheckIcon,
  Info as InfoIcon,
  Stars as StarsIcon
} from '@mui/icons-material';

const FeatureCard = ({ title, description, limit, isPremium }) => (
  <Card 
    sx={{ 
      height: '100%',
      backgroundColor: isPremium ? 'rgba(33, 150, 243, 0.05)' : 'transparent',
      border: '1px solid',
      borderColor: isPremium ? 'primary.main' : 'divider',
      transition: 'all 0.3s ease',
      '&:hover': {
        transform: 'translateY(-5px)',
        boxShadow: isPremium ? '0 8px 24px rgba(33, 150, 243, 0.15)' : '0 8px 24px rgba(0,0,0,0.1)',
      }
    }}
  >
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
        {isPremium && (
          <Tooltip title="Premium Feature">
            <StarsIcon sx={{ color: 'primary.main' }} />
          </Tooltip>
        )}
      </Box>
      <Typography color="text.secondary" paragraph>
        {description}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <InfoIcon sx={{ fontSize: '1rem', color: 'text.secondary' }} />
        <Typography variant="body2" color="text.secondary">
          Limit: {limit}
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

const SubscriptionFeatures = () => {
  const { getUserSubscription, subscriptionLevels, user } = useUser();
  const navigate = useNavigate();
  const currentSubscription = getUserSubscription();
  const features = subscriptionLevels[currentSubscription].features;

  return (
    <Box>
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        mb: 4 
      }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Your Subscription Features
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            {currentSubscription === 'premium' 
              ? 'You have access to all premium features'
              : 'Upgrade to premium for advanced features'}
          </Typography>
        </Box>
        {currentSubscription === 'free' && (
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
        )}
      </Box>

      <Grid container spacing={3}>
        {Object.entries(features).map(([key, feature]) => (
          <Grid item xs={12} sm={6} md={4} key={key}>
            <FeatureCard
              title={key.charAt(0).toUpperCase() + key.slice(1)}
              description={feature.description}
              limit={feature.limit}
              isPremium={currentSubscription === 'premium'}
            />
          </Grid>
        ))}
      </Grid>

      {!user && (
        <Box sx={{ 
          mt: 4, 
          p: 3, 
          borderRadius: 2,
          bgcolor: 'rgba(33, 150, 243, 0.05)',
          border: '1px solid',
          borderColor: 'primary.main',
        }}>
          <Typography variant="h6" gutterBottom>
            Get Started with DocuMorph AI
          </Typography>
          <Typography paragraph>
            Create an account to access more features and start transforming your documents today.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={() => navigate('/register')}
              sx={{
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
              }}
            >
              Create Account
            </Button>
            <Button
              variant="outlined"
              color="primary"
              onClick={() => navigate('/login')}
            >
              Login
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default SubscriptionFeatures;