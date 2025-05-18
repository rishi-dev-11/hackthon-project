import React from 'react';
import { motion as Motion } from 'framer-motion';
import { Alert, Box } from '@mui/material';

const AnimatedNotification = ({ message, severity = 'info' }) => {
  return (
    <Motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <Box sx={{ position: 'fixed', top: 20, right: 20, zIndex: 2000 }}>
        <Alert 
          severity={severity}
          sx={{ 
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            borderRadius: 2
          }}
        >
          {message}
        </Alert>
      </Box>
    </Motion.div>
  );
};

export default AnimatedNotification; 