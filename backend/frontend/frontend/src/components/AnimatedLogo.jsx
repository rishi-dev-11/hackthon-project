import React from 'react';
import { Box, Typography } from '@mui/material';
import { motion as Motion } from 'framer-motion';
import hackLogo from '../assets/hack_logo.png';

const AnimatedLogo = () => {
  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center',
      justifyContent: { xs: 'center', sm: 'flex-start' },
      px: 2,
      py: 1,
      ml: '280px' // Width of HamburgerMenu
    }}>
      <Motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <img 
          src={hackLogo} 
          alt="DocuMorph Logo" 
          style={{ 
            height: '40px',
            marginRight: '12px'
          }} 
        />
      </Motion.div>
      <Motion.div
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: { xs: '1.25rem', md: '1.5rem' }
          }}
        >
          DocuMorph AI
        </Typography>
      </Motion.div>
    </Box>
  );
};

export default AnimatedLogo;