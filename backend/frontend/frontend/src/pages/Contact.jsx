import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper,
  TextField,
  Button,
  Grid,
  Snackbar,
  Alert
} from '@mui/material';
import { motion as Motion } from 'framer-motion';
import { pageTransition, staggerContainer, fadeInUp } from '../styles/animations';

function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [showAlert, setShowAlert] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    setShowAlert(true);
  };

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
              Contact Us
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
              <Grid container spacing={4}>
                <Grid item xs={12} md={6}>
                  <Motion.div variants={fadeInUp}>
                    <Typography variant="h5" gutterBottom color="primary">
                      Get in Touch
                    </Typography>
                    <Typography variant="body1" paragraph>
                      Have questions about DocuMorph AI? We're here to help. 
                      Fill out the form and we'll get back to you as soon as possible.
                    </Typography>
                    <Typography variant="body1" paragraph>
                      Email: support@documorph.ai
                    </Typography>
                  </Motion.div>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Motion.div variants={fadeInUp}>
                    <form onSubmit={handleSubmit}>
                      <TextField
                        fullWidth
                        label="Name"
                        margin="normal"
                        required
                        value={formData.name}
                        onChange={(e) => setFormData({...formData, name: e.target.value})}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            '&:hover fieldset': {
                              borderColor: 'primary.main',
                            },
                          },
                        }}
                      />
                      <TextField
                        fullWidth
                        label="Email"
                        type="email"
                        margin="normal"
                        required
                        value={formData.email}
                        onChange={(e) => setFormData({...formData, email: e.target.value})}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            '&:hover fieldset': {
                              borderColor: 'primary.main',
                            },
                          },
                        }}
                      />
                      <TextField
                        fullWidth
                        label="Message"
                        multiline
                        rows={4}
                        margin="normal"
                        required
                        value={formData.message}
                        onChange={(e) => setFormData({...formData, message: e.target.value})}
                        sx={{
                          '& .MuiOutlinedInput-root': {
                            '&:hover fieldset': {
                              borderColor: 'primary.main',
                            },
                          },
                        }}
                      />
                      <Motion.div
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Button 
                          type="submit"
                          variant="contained"
                          fullWidth
                          sx={{ 
                            mt: 2,
                            py: 1.5,
                            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                            boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                            '&:hover': {
                              background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
                            }
                          }}
                        >
                          Send Message
                        </Button>
                      </Motion.div>
                    </form>
                  </Motion.div>
                </Grid>
              </Grid>
            </Motion.div>
          </Paper>
        </Box>
      </Container>

      <Snackbar 
        open={showAlert} 
        autoHideDuration={6000} 
        onClose={() => setShowAlert(false)}
      >
        <Alert 
          severity="success" 
          onClose={() => setShowAlert(false)}
          sx={{ 
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            borderRadius: 2
          }}
        >
          Message sent successfully!
        </Alert>
      </Snackbar>
    </Motion.div>
  );
}

export default Contact; 