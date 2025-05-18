import React from 'react';
import { motion as Motion } from 'framer-motion';
import { Box, Typography } from '@mui/material';

const AnimatedProgress = ({ progress, label }) => {
  return (
    <Box sx={{ width: '100%', mb: 2 }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        mb: 1
      }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {progress}%
        </Typography>
      </Box>
      <Box sx={{ 
        height: 8, 
        backgroundColor: 'rgba(0,0,0,0.1)',
        borderRadius: 4,
        overflow: 'hidden'
      }}>
        <Motion.div
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, #2196f3, #21CBF3)',
            borderRadius: 4
          }}
        />
      </Box>
    </Box>
  );
};

export default AnimatedProgress; 