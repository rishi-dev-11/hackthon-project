import React from 'react';
import { motion as Motion } from 'framer-motion';
import { Box } from '@mui/material';
import DocumentScannerIcon from '@mui/icons-material/DocumentScanner';

const AnimatedScannerIcon = ({ size = 24, color = 'primary' }) => {
  return (
    <Box sx={{ position: 'relative', width: size, height: size }}>
      <DocumentScannerIcon 
        sx={{ 
          fontSize: size,
          color: `${color}.main`,
          position: 'relative',
          zIndex: 1
        }} 
      />
      <Motion.div
        initial={{ y: 0 }}
        animate={{ 
          y: [0, size, 0],
          opacity: [0, 0.5, 0]
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "linear"
        }}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: 'linear-gradient(90deg, transparent, #2196f3, transparent)',
          zIndex: 2
        }}
      />
    </Box>
  );
};

export default AnimatedScannerIcon; 