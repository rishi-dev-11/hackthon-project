import React, { useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Button,
  Paper,
  Divider,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Stars as StarsIcon,
  Check as CheckIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSubscription } from '../context/SubscriptionContext';
import { updateUserSubscription } from '../services/api';
import gsap from 'gsap';

const PlanFeature = ({ name, included, description }) => (
  <ListItem disableGutters>
    <ListItemIcon sx={{ minWidth: 40 }}>
      {included ? (
        <CheckIcon color="success" />
      ) : (
        <CloseIcon color="error" />
      )}
    </ListItemIcon>
    <ListItemText 
      primary={name}
      secondary={description}
      primaryTypographyProps={{
        fontWeight: included ? 500 : 400,
        color: included ? 'text.primary' : 'text.secondary'
      }}
    />
  </ListItem>
);

const SubscriptionPlan = ({ plan, isPopular, onSelect }) => {
  const planRef = useRef(null);
  
  useEffect(() => {
    // GSAP animation replacing framer-motion
    gsap.fromTo(
      planRef.current,
      { y: 20, opacity: 0 },
      { 
        y: 0, 
        opacity: 1, 
        duration: 0.5, 
        delay: plan.delay,
        ease: "power2.out"
      }
    );
  }, [plan.delay]);

  const features = plan.features.map((feature, index) => (
    <PlanFeature key={index} {...feature} />
  ));

  return (
    <div ref={planRef}>
      <Paper
        elevation={0}
        sx={{
          p: 3,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          borderRadius: 2,
          border: '1px solid',
          borderColor: isPopular ? 'primary.main' : 'divider',
          backgroundColor: isPopular ? 'primary.light' : 'background.paper',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
          }
        }}
      >
        {isPopular && (
          <Chip
            label="Most Popular"
            color="primary"
            icon={<StarsIcon />}
            sx={{
              position: 'absolute',
              top: -12,
              right: 24,
            }}
          />
        )}

        <Typography variant="h5" gutterBottom fontWeight="bold">
          {plan.name}
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="h3" component="span" fontWeight="bold">
            ${plan.price}
          </Typography>
          <Typography variant="subtitle1" component="span" color="text.secondary">
            /month
          </Typography>
        </Box>

        <Typography color="text.secondary" paragraph>
          {plan.description}
        </Typography>

        <Divider sx={{ my: 2 }} />

        <List sx={{ mb: 2, flexGrow: 1 }}>
          {features}
        </List>

        <Button
          variant={isPopular ? "contained" : "outlined"}
          size="large"
          onClick={() => onSelect(plan)}
          startIcon={isPopular && <StarsIcon />}
          sx={{
            mt: 'auto',
            py: 1.5,
            ...(isPopular && {
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
            })
          }}
        >
          {plan.buttonText}
        </Button>
      </Paper>
    </div>
  );
};

const Upgrade = () => {
  const navigate = useNavigate();
  const containerRef = useRef(null);
  const { userTier, isDevMode, switchPlan } = useSubscription();
  const [upgradeStatus, setUpgradeStatus] = React.useState({ message: '', type: '' });

  useEffect(() => {
    // GSAP animation for the container
    gsap.fromTo(
      containerRef.current,
      { opacity: 0 },
      { opacity: 1, duration: 0.8, ease: "power1.out" }
    );
  }, []);

  const handlePlanSelect = async (plan) => {
    try {
      setUpgradeStatus({ message: '', type: '' });
      
      if (isDevMode) {
        // In dev mode, use the switchPlan function from context
        switchPlan(plan.name.toLowerCase());
        setUpgradeStatus({ 
          message: `Successfully switched to ${plan.name} plan in development mode`, 
          type: 'success' 
        });
      } else {
        // In production, call the API to update subscription
        await updateUserSubscription(plan.name.toLowerCase());
        setUpgradeStatus({ 
          message: 'Subscription updated successfully', 
          type: 'success' 
        });
      }
      
      // Navigate back after a short delay
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    } catch (error) {
      console.error('Error upgrading subscription:', error);
      setUpgradeStatus({ 
        message: `Failed to update subscription: ${error.message || 'Unknown error'}`, 
        type: 'error' 
      });
    }
  };

  const plans = [
    {
      name: 'Free',
      price: 0,
      description: 'Get started with basic document processing',
      delay: 0,
      buttonText: userTier === 'free' ? 'Current Plan' : 'Switch to Free',
      features: [
        { name: 'Basic Templates', included: true, description: 'Access to standard templates' },
        { name: 'Document Processing', included: true, description: '5 documents per month' },
        { name: 'Basic OCR', included: true, description: 'Text extraction from images' },
        { name: 'Storage', included: true, description: '500MB storage space' },
        { name: 'Advanced Templates', included: false, description: 'Premium template collection' },
        { name: 'Analytics', included: false, description: 'Document processing insights' },
      ]
    },
    {
      name: 'Premium',
      price: 9.99,
      description: 'Perfect for professionals and teams',
      delay: 0.1,
      buttonText: userTier === 'premium' ? 'Current Plan' : 'Upgrade Now',
      features: [
        { name: 'All Basic Features', included: true, description: 'Everything in Free plan' },
        { name: 'Advanced Templates', included: true, description: 'Full template collection' },
        { name: 'Unlimited Processing', included: true, description: 'No monthly limits' },
        { name: 'Advanced OCR', included: true, description: 'Enhanced text recognition' },
        { name: 'Analytics Dashboard', included: true, description: 'Detailed insights' },
        { name: 'Priority Support', included: true, description: '24/7 premium support' },
      ]
    }
  ];

  return (
    <Container ref={containerRef} maxWidth="lg" sx={{ py: 6 }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" gutterBottom fontWeight="bold">
          Choose Your Plan
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Unlock premium features and transform your documents like never before
        </Typography>
      </Box>

      {isDevMode && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Developer Mode: You can freely switch between plans for testing.
        </Alert>
      )}
      
      {upgradeStatus.message && (
        <Alert severity={upgradeStatus.type} sx={{ mb: 3 }}>
          {upgradeStatus.message}
        </Alert>
      )}

      <Grid container spacing={4} alignItems="stretch">
        {plans.map(plan => (
          <Grid gridColumn={{ xs: 'span 12', md: 'span 6' }} key={plan.name}>
            <SubscriptionPlan
              plan={plan}
              isPopular={plan.name === 'Premium'}
              onSelect={handlePlanSelect}
            />
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          All plans include our core features. Upgrade anytime to access premium features.
        </Typography>
      </Box>
    </Container>
  );
};

export default Upgrade;