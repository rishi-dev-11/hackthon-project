import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import { motion as Motion } from 'framer-motion';
import { pageTransition, staggerContainer, fadeInUp } from '../styles/animations';

function Terms() {
  return (
    <Motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageTransition}
    >
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Motion.div variants={fadeInUp}>
            <Typography 
              variant="h3" 
              gutterBottom 
              align="center" 
              sx={{ 
                fontWeight: 'bold',
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                backgroundClip: 'text',
                textFillColor: 'transparent',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              Terms of Service
            </Typography>
          </Motion.div>
          
          <Paper 
            elevation={0} 
            sx={{ 
              p: 4, 
              my: 4, 
              borderRadius: 4,
              background: 'rgba(255, 255, 255, 0.9)',
              backdropFilter: 'blur(10px)',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
              transition: 'transform 0.3s ease',
              '&:hover': {
                transform: 'translateY(-5px)'
              }
            }}
          >
            <Motion.div variants={staggerContainer}>
              <Motion.div variants={fadeInUp}>
                <Typography variant="h5" gutterBottom color="primary">
                  1. Acceptance of Terms
                </Typography>
                <Typography variant="body1" paragraph>
                  By accessing and using DocuMorph AI, you agree to be bound by these Terms of Service.
                </Typography>
              </Motion.div>

              <Motion.div variants={fadeInUp}>
                <Typography variant="h5" gutterBottom color="primary" sx={{ mt: 4 }}>
                  2. Use of Service
                </Typography>
                <Typography variant="body1" paragraph>
                  DocuMorph AI provides AI-powered document formatting services. You agree to use the service 
                  only for lawful purposes and in accordance with these Terms.
                </Typography>
              </Motion.div>

              <Motion.div variants={fadeInUp}>
                <Typography variant="h5" gutterBottom color="primary" sx={{ mt: 4 }}>
                  3. User Responsibilities
                </Typography>
                <List>
                  {[
                    {
                      primary: "Provide accurate information",
                      secondary: "You must provide accurate and complete information when using our service."
                    },
                    {
                      primary: "Maintain account security",
                      secondary: "You are responsible for maintaining the security of your account."
                    },
                    {
                      primary: "Comply with laws",
                      secondary: "You must comply with all applicable laws and regulations."
                    }
                  ].map((item, index) => (
                    <Motion.div
                      key={item.primary}
                      variants={fadeInUp}
                      transition={{ delay: 0.3 + index * 0.2 }}
                    >
                      <ListItem>
                        <ListItemText 
                          primary={item.primary}
                          secondary={item.secondary}
                          sx={{
                            '& .MuiListItemText-primary': {
                              color: 'primary.main',
                              fontWeight: 500
                            }
                          }}
                        />
                      </ListItem>
                    </Motion.div>
                  ))}
                </List>
              </Motion.div>

              <Motion.div variants={fadeInUp}>
                <Typography variant="h5" gutterBottom color="primary" sx={{ mt: 4 }}>
                  4. Privacy
                </Typography>
                <Typography variant="body1" paragraph>
                  Your privacy is important to us. Please review our Privacy Policy to understand how we 
                  collect and use your information.
                </Typography>
              </Motion.div>
            </Motion.div>
          </Paper>
        </Box>
      </Container>
    </Motion.div>
  );
}

export default Terms;
