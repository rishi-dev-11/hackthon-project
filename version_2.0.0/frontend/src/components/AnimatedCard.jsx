import React from 'react';
import { motion as Motion } from 'framer-motion';
import { 
  Card, 
  CardContent, 
  CardActions, 
  Button, 
  Box,
  Typography 
} from '@mui/material';

const AnimatedCard = ({ 
  title, 
  description, 
  icon: Icon, 
  onClick, 
  delay = 0 
}) => {
  return (
    <Motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      whileHover={{ 
        scale: 1.02,
        transition: { duration: 0.2 }
      }}
    >
      <Card sx={{ 
        height: '100%',
        background: 'linear-gradient(145deg, #ffffff 0%, #f5f7fa 100%)',
        borderRadius: 4,
        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        overflow: 'hidden',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #2196f3, #21CBF3)',
          transform: 'scaleX(0)',
          transition: 'transform 0.3s ease'
        },
        '&:hover::before': {
          transform: 'scaleX(1)'
        }
      }}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 2 
          }}>
            <Motion.div
              whileHover={{ 
                rotate: 360,
                scale: 1.2,
                transition: { duration: 0.5 }
              }}
            >
              {Icon && <Icon sx={{ 
                fontSize: 32,
                color: 'primary.main',
                mr: 2
              }} />}
            </Motion.div>
            <Motion.div
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: delay + 0.2 }}
            >
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {title}
              </Typography>
            </Motion.div>
          </Box>
          <Motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: delay + 0.3 }}
          >
            <Typography variant="body1" color="text.secondary">
              {description}
            </Typography>
          </Motion.div>
        </CardContent>
        <CardActions sx={{ p: 2, pt: 0 }}>
          <Motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{ width: '100%' }}
          >
            <Button 
              variant="contained" 
              fullWidth
              onClick={onClick}
              sx={{ 
                py: 1.5,
                fontSize: '1rem',
                fontWeight: 600,
                borderRadius: 2,
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)'
              }}
            >
              Get Started
            </Button>
          </Motion.div>
        </CardActions>
      </Card>
    </Motion.div>
  );
};

export default AnimatedCard; 