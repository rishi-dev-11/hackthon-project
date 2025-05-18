import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import SuccessIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';

const StatusCard = ({ status, title, message, isLoading }) => {
  const getStatusIcon = () => {
    if (isLoading) {
      return <CircularProgress size={24} />;
    }
    
    switch (status) {
      case 'success':
        return <SuccessIcon color="success" fontSize="medium" />;
      case 'warning':
        return <WarningIcon color="warning" fontSize="medium" />;
      case 'error':
        return <ErrorIcon color="error" fontSize="medium" />;
      case 'info':
      default:
        return <InfoIcon color="info" fontSize="medium" />;
    }
  };
  
  const getStatusColor = () => {
    switch (status) {
      case 'success':
        return 'success.main';
      case 'warning':
        return 'warning.main';
      case 'error':
        return 'error.main';
      case 'info':
      default:
        return 'info.main';
    }
  };
  
  return (
    <Card 
      sx={{ 
        mb: 2,
        borderLeft: 4,
        borderColor: getStatusColor(),
        boxShadow: 1
      }}
    >
      <CardContent sx={{ display: 'flex', alignItems: 'flex-start', p: 2 }}>
        <Box sx={{ mr: 2, mt: 0.5 }}>
          {getStatusIcon()}
        </Box>
        <Box>
          <Typography variant="subtitle1" component="h3" fontWeight="500">
            {title}
          </Typography>
          {message && (
            <Typography variant="body2" color="text.secondary">
              {message}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default StatusCard;
