import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Check as CheckIcon,
  Close as CloseIcon,
  Stars as StarsIcon,
} from '@mui/icons-material';
import { useUser } from '../hooks/useUser';
import { useNavigate } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';

const PlanComparison = () => {
  const { getUserSubscription } = useUser();
  const navigate = useNavigate();
  const currentPlan = getUserSubscription();

  const featureCategories = [
    {
      name: 'Document Processing',
      features: [
        {
          name: 'Basic Document Upload',
          free: true,
          premium: true,
        },
        {
          name: 'OCR Text Extraction',
          free: 'Basic',
          premium: 'Advanced',
        },
        {
          name: 'Template Application',
          free: 'Basic Templates',
          premium: 'All Templates',
        }
      ]
    },
    {
      name: 'Templates & Formatting',
      features: [
        {
          name: 'Template Creation',
          free: 'Basic',
          premium: 'Advanced + Custom',
        },
        {
          name: 'Brand Integration',
          free: false,
          premium: true,
        },
        {
          name: 'Style Customization',
          free: 'Limited',
          premium: 'Full Control',
        }
      ]
    },
    {
      name: 'Language & Translation',
      features: [
        {
          name: 'Multi-language Support',
          free: 'Basic',
          premium: 'Advanced',
        },
        {
          name: 'Auto Language Detection',
          free: false,
          premium: true,
        },
        {
          name: 'Template Translation',
          free: false,
          premium: true,
        }
      ]
    },
    {
      name: 'Collaboration',
      features: [
        {
          name: 'Document Sharing',
          free: 'Basic',
          premium: 'Advanced',
        },
        {
          name: 'Version Control',
          free: '2 Versions',
          premium: 'Unlimited',
        },
        {
          name: 'Team Features',
          free: false,
          premium: true,
        }
      ]
    }
  ];

  const renderFeatureValue = (value) => {
    if (typeof value === 'boolean') {
      return value ? (
        <CheckIcon sx={{ color: 'success.main' }} />
      ) : (
        <CloseIcon sx={{ color: 'error.main' }} />
      );
    }
    return value;
  };

  return (
    <Box sx={{ py: 4 }}>
      <Motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom align="center">
          Choose Your Plan
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 6 }}>
          Compare our plans and choose the best one for your needs
        </Typography>

        <Grid container spacing={4} justifyContent="center">
          {['free', 'premium'].map((plan) => (
            <Grid item xs={12} md={6} key={plan}>
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  height: '100%',
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: plan === 'premium' ? 'primary.main' : 'divider',
                  bgcolor: plan === 'premium' ? 'rgba(33, 150, 243, 0.05)' : 'transparent',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                {plan === currentPlan && (
                  <Chip
                    label="Current Plan"
                    color="primary"
                    size="small"
                    sx={{
                      position: 'absolute',
                      top: 16,
                      right: 16,
                    }}
                  />
                )}
                <Typography variant="h5" gutterBottom>
                  {plan === 'premium' ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      Premium Plan
                      <StarsIcon sx={{ color: 'primary.main' }} />
                    </Box>
                  ) : (
                    'Free Plan'
                  )}
                </Typography>
                <Typography variant="h3" gutterBottom>
                  {plan === 'premium' ? '$9.99' : '$0'}
                  <Typography component="span" variant="body1" color="text.secondary">
                    /month
                  </Typography>
                </Typography>

                {plan !== currentPlan && plan === 'premium' && (
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={() => navigate('/upgrade')}
                    sx={{
                      mt: 2,
                      mb: 4,
                      background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                      boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                    }}
                  >
                    Upgrade Now
                  </Button>
                )}

                {featureCategories.map((category) => (
                  <Box key={category.name} sx={{ mt: 3 }}>
                    <Typography
                      variant="subtitle1"
                      color="primary"
                      gutterBottom
                      sx={{ fontWeight: 'bold' }}
                    >
                      {category.name}
                    </Typography>
                    {category.features.map((feature) => (
                      <Box
                        key={feature.name}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          py: 1,
                          borderBottom: '1px solid',
                          borderColor: 'divider',
                        }}
                      >
                        <Typography variant="body2">
                          {feature.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {renderFeatureValue(feature[plan])}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                ))}
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Motion.div>
    </Box>
  );
};

export default PlanComparison;