import React from 'react';
import { motion as Motion } from 'framer-motion';
import { Box } from '@mui/material';

const AnimatedIcon = ({ icon: Icon, color = 'primary', size = 24 }) => {
  return (
    <Motion.div
      whileHover={{ 
        rotate: 360,
        scale: 1.2,
        transition: { duration: 0.5, ease: "easeInOut" }
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: `${color}.main`
      }}>
        {Icon && <Icon sx={{ fontSize: size }} />}
      </Box>
    </Motion.div>
  );
};

export default AnimatedIcon; 