import React from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Typography, 
  Link,
  Divider
} from '@mui/material';
import { motion as Motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { pageTransition, hoverScale, fadeInUp } from '../styles/animations'; // Ensure fadeInUp is also imported

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const navigate = useNavigate();

  const footerLinks = [
    { name: "About", path: "/about" },
    { name: "Contact Us", path: "/contact" },
    { name: "Terms of Service", path: "/terms" }
  ];

  return (
    <Motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageTransition}
    >
      <Box
        component="footer"
        sx={{
          py: 4,
          px: 2,
          mt: 'auto',
          backgroundColor: 'background.paper',
          borderTop: '1px solid',
          borderColor: 'divider',
          boxShadow: '0 -4px 6px -1px rgba(0, 0, 0, 0.1)',
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '4px',
            background: 'linear-gradient(90deg, #2196f3, #21CBF3)',
            opacity: 0.5
          }
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={2} alignItems="center" justifyContent="center">
            <Grid item xs={12} md={4}>
              <Motion.div
                variants={fadeInUp}
                transition={{ delay: 0.2 }}
              >
                <Typography 
                  variant="h6" 
                  color="primary" 
                  gutterBottom 
                  align="center"
                  sx={{
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    backgroundClip: 'text',
                    textFillColor: 'transparent',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                  }}
                >
                  DocuMorph AI
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  paragraph 
                  align="center"
                  sx={{ maxWidth: '400px', mx: 'auto' }}
                >
                  Transform your documents with AI-powered formatting
                </Typography>
              </Motion.div>
            </Grid>
          </Grid>

          <Divider sx={{ 
            my: 2,
            borderColor: 'rgba(0, 0, 0, 0.1)',
            '&::before, &::after': {
              borderColor: 'rgba(0, 0, 0, 0.1)'
            }
          }} />

          <Grid container spacing={2} alignItems="center" justifyContent="center">
            <Grid item xs={12}>
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                gap: 4,
                flexWrap: 'wrap'
              }}>
                {footerLinks.map((link, index) => (
                  <Motion.div
                    key={link.name}
                    variants={fadeInUp}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    whileHover={hoverScale.whileHover}
                    whileTap={hoverScale.whileTap}
                  >
                    <Link
                      component="button"
                      onClick={() => navigate(link.path)}
                      sx={{
                        color: 'text.secondary',
                        textDecoration: 'none',
                        fontSize: '1rem',
                        fontWeight: 500,
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: 'primary.main',
                          textDecoration: 'underline'
                        }
                      }}
                    >
                      {link.name}
                    </Link>
                  </Motion.div>
                ))}
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Motion.div
                variants={fadeInUp}
                transition={{ delay: 0.5 }}
              >
                <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  align="center"
                  sx={{ mt: 2 }}
                >
                  © {currentYear} DocuMorph AI. All rights reserved.
                </Typography>
              </Motion.div>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Motion.div>
  );
};

export default Footer;
