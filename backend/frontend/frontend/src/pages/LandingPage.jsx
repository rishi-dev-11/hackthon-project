import React, { useState } from 'react';
import { Box, Button, Container, Typography, Grid, Card, CardContent, IconButton, Dialog, DialogContent, DialogActions, TextField, Alert } from '@mui/material';
import { motion, useAnimation } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../hooks/useUser';
import { 
  AutoAwesome, 
  Speed, 
  Security, 
  CloudUpload,
  ArrowForward,
  GitHub,
  LinkedIn,
  Twitter,
  Description,
  Psychology,
} from '@mui/icons-material';

// Create a motion component using motion.create() to avoid deprecation warnings
const MotionBox = motion(Box);
const MotionButton = motion(Button);
const MotionIcon = motion(Box);
const MotionContainer = motion(Container);
const MotionCard = motion(Card);
const MotionTypography = motion(Typography);
const MotionGrid = motion(Grid);

const features = [
  {
    icon: <AutoAwesome sx={{ fontSize: 40 }} />,
    title: "AI-Powered Conversion",
    description: "Transform your documents with advanced AI technology"
  },
  {
    icon: <Speed sx={{ fontSize: 40 }} />,
    title: "Lightning Fast",
    description: "Process documents in seconds, not minutes"
  },
  {
    icon: <Security sx={{ fontSize: 40 }} />,
    title: "Secure & Private",
    description: "Your documents are encrypted and secure"
  },
  {
    icon: <CloudUpload sx={{ fontSize: 40 }} />,
    title: "Easy Upload",
    description: "Drag and drop your files for instant processing"
  }
];

const particles = Array.from({ length: 20 }, (_, i) => ({
  id: i,
  size: Math.random() * 10 + 3,
  x: Math.random() * 100,
  y: Math.random() * 100,
  duration: Math.random() * 20 + 10,
  delay: Math.random() * 5
}));

// Component for user onboarding
const OnboardingDialog = ({ open, onClose, onComplete }) => {
  const [step, setStep] = useState(0);
  const [userData, setUserData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    purpose: '',
    interests: []
  });
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const purposes = [
    { value: 'Student', label: 'Student', description: 'Use for academic papers and assignments' },
    { value: 'Content Creator', label: 'Content Creator', description: 'Create professional content for your audience' },
    { value: 'Researcher', label: 'Researcher', description: 'Advanced research papers and publications' },
    { value: 'Business Professional', label: 'Business Professional', description: 'Business reports and presentations' },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setUserData(prev => ({ ...prev, [name]: value }));
  };

  const handlePurposeSelect = (purpose) => {
    setUserData(prev => ({ ...prev, purpose }));
  };

  const validateStep = () => {
    setError('');
    switch (step) {
      case 0: // Name step
        if (!userData.name.trim()) {
          setError('Please enter your name');
          return false;
        }
        break;
      case 1: // Email/password step
        if (!userData.email.trim()) {
          setError('Please enter your email');
          return false;
        }
        if (!userData.password) {
          setError('Please enter a password');
          return false;
        }
        if (userData.password !== userData.confirmPassword) {
          setError('Passwords do not match');
          return false;
        }
        if (userData.password.length < 6) {
          setError('Password must be at least 6 characters');
          return false;
        }
        break;
      case 2: // Purpose step
        if (!userData.purpose) {
          setError('Please select a purpose');
          return false;
        }
        break;
    }
    return true;
  };

  const handleNext = () => {
    if (validateStep()) {
      if (step < 2) {
        setStep(step + 1);
      } else {
        handleSubmit();
      }
    }
  };

  const handleBack = () => {
    if (step > 0) {
      setStep(step - 1);
    } else {
      onClose();
    }
  };

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true);
      const response = await registerUser({
        name: userData.name,
        email: userData.email,
        password: userData.password,
        purpose: userData.purpose
      });
      
      // Store the token received after registration
      localStorage.setItem('token', response.access_token);
      
      onComplete(response);
    } catch (error) {
      console.error('Registration failed:', error);
      setError(error.response?.data?.detail || 'Registration failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStepContent = () => {
    switch (step) {
      case 0:
        return (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h5" gutterBottom>Welcome to DocuMorph AI</Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Let's start by getting to know you. What's your name?
            </Typography>
            <TextField
              fullWidth
              name="name"
              label="Your Name"
              value={userData.name}
              onChange={handleChange}
              variant="outlined"
              margin="normal"
              autoFocus
            />
          </Box>
        );
      case 1:
        return (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="h5" gutterBottom>Create your account</Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Set up your account to save your documents and preferences
            </Typography>
            <TextField
              fullWidth
              name="email"
              type="email"
              label="Email Address"
              value={userData.email}
              onChange={handleChange}
              variant="outlined"
              margin="normal"
            />
            <TextField
              fullWidth
              name="password"
              type="password"
              label="Password"
              value={userData.password}
              onChange={handleChange}
              variant="outlined"
              margin="normal"
            />
            <TextField
              fullWidth
              name="confirmPassword"
              type="password"
              label="Confirm Password"
              value={userData.confirmPassword}
              onChange={handleChange}
              variant="outlined"
              margin="normal"
            />
          </Box>
        );
      case 2:
        return (
          <Box sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom textAlign="center">How will you use DocuMorph?</Typography>
            <Typography variant="body1" color="text.secondary" paragraph textAlign="center">
              We'll customize your experience based on your needs
            </Typography>
            <Grid container spacing={2} sx={{ mt: 2 }}>
              {purposes.map((purpose) => (
                <Grid item xs={12} sm={6} key={purpose.value}>
                  <Card 
                    onClick={() => handlePurposeSelect(purpose.value)}
                    sx={{ 
                      p: 2, 
                      cursor: 'pointer',
                      border: userData.purpose === purpose.value ? '2px solid' : '1px solid',
                      borderColor: userData.purpose === purpose.value ? 'primary.main' : 'divider',
                      bgcolor: userData.purpose === purpose.value ? 'rgba(33, 150, 243, 0.05)' : 'transparent',
                      transition: 'all 0.2s',
                      '&:hover': {
                        borderColor: 'primary.light',
                        transform: 'translateY(-5px)',
                        boxShadow: 2
                      }
                    }}
                  >
                    <Typography variant="h6">{purpose.label}</Typography>
                    <Typography variant="body2" color="text.secondary">{purpose.description}</Typography>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <Dialog
      open={open}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          boxShadow: 24
        }
      }}
    >
      <DialogContent>
        {renderStepContent()}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 3 }}>
        <Button onClick={handleBack}>
          {step === 0 ? 'Skip' : 'Back'}
        </Button>
        <Button 
          variant="contained" 
          onClick={handleNext}
          disabled={isSubmitting}
          endIcon={isSubmitting ? <CircularProgress size={16} /> : null}
        >
          {step === 2 ? 'Get Started' : 'Next'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const LandingPage = () => {
  const navigate = useNavigate();
  const { user } = useUser();
  const controls = useAnimation();
  const [onboardingOpen, setOnboardingOpen] = useState(false);

  React.useEffect(() => {
    controls.start({
      opacity: [0.4, 1, 0.4],
      scale: [1, 1.2, 1],
      transition: { 
        duration: 5, 
        repeat: Infinity,
        repeatType: "reverse" 
      }
    });
  }, [controls]);

  const handleGetStarted = () => {
    setOnboardingOpen(true);
  };

  const handleOnboardingComplete = (userData) => {
    setOnboardingOpen(false);
    navigate('/dashboard');
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <Box sx={{ 
      width: '100%',
      minHeight: '100vh',
      background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
      overflowX: 'hidden',
      position: 'relative'
    }}>
      {/* Animated Background Particles */}
      {particles.map((particle) => (
        <MotionBox
          key={particle.id}
          initial={{ 
            x: `${particle.x}%`, 
            y: `${particle.y}%`, 
            opacity: 0.3,
            scale: 0.8
          }}
          animate={{ 
            y: [`${particle.y}%`, `${(particle.y + 20) % 100}%`],
            opacity: [0.3, 0.7, 0.3],
            scale: [0.8, 1.2, 0.8]
          }}
          transition={{ 
            duration: particle.duration,
            repeat: Infinity,
            repeatType: "reverse",
            delay: particle.delay
          }}
          sx={{
            position: 'absolute',
            width: particle.size,
            height: particle.size,
            borderRadius: '50%',
            background: 'white',
            filter: 'blur(1px)',
            zIndex: 0
          }}
        />
      ))}

      <Container maxWidth="lg" sx={{ 
        py: { xs: 4, md: 6 },
        px: { xs: 2, sm: 3, md: 4 },
        position: 'relative',
        zIndex: 1
      }}>
        <Grid container spacing={4} alignItems="center" sx={{ minHeight: '90vh' }}>
          <Grid item xs={12} md={6}>
            <MotionBox
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <Typography 
                variant="h2" 
                component="h1" 
                gutterBottom
                sx={{ 
                  fontWeight: 'bold',
                  textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
                  color: 'white',
                  fontSize: { xs: '2.5rem', sm: '3rem', md: '3.5rem' },
                  lineHeight: 1.2
                }}
              >
                Transform Your Documents with AI Power
              </Typography>
              <Typography 
                variant="h5" 
                sx={{ 
                  mb: 4,
                  opacity: 0.9,
                  color: 'white',
                  fontSize: { xs: '1.1rem', sm: '1.3rem', md: '1.5rem' },
                  lineHeight: 1.6
                }}
              >
                DocuMorph AI is your intelligent document transformation platform that revolutionizes how professionals handle documents. From academic papers to business proposals, our AI-powered engine analyzes, formats, and enhances your documents with precision and style. Experience the future of document processing with smart templates, multilingual support, and enterprise-grade security.
              </Typography>

              <Box sx={{ mt: 4, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                {user ? (
                  <MotionButton
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/dashboard')}
                    sx={{
                      backgroundColor: 'white',
                      color: '#2196F3',
                      px: { xs: 3, md: 4 },
                      py: 1.5,
                      fontSize: { xs: '1rem', md: '1.1rem' },
                      fontWeight: 600,
                      '&:hover': {
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                      },
                    }}
                  >
                    Go to Dashboard
                  </MotionButton>
                ) : (
                  <>
                    <MotionButton
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      variant="contained"
                      size="large"
                      onClick={handleGetStarted}
                      sx={{
                        backgroundColor: 'white',
                        color: '#2196F3',
                        px: { xs: 3, md: 4 },
                        py: 1.5,
                        fontSize: { xs: '1rem', md: '1.1rem' },
                        fontWeight: 600,
                        '&:hover': {
                          backgroundColor: 'rgba(255, 255, 255, 0.9)',
                          transform: 'translateY(-2px)',
                          boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                        },
                      }}
                    >
                      Get Started
                    </MotionButton>
                    <MotionButton
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      variant="outlined"
                      size="large"
                      onClick={() => navigate('/about')}
                      sx={{
                        borderColor: 'white',
                        color: 'white',
                        px: { xs: 3, md: 4 },
                        py: 1.5,
                        fontSize: { xs: '1rem', md: '1.1rem' },
                        fontWeight: 600,
                        '&:hover': {
                          borderColor: 'white',
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          transform: 'translateY(-2px)',
                          boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                        },
                      }}
                    >
                      Learn More
                    </MotionButton>
                  </>
                )}
              </Box>
            </MotionBox>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              position: 'relative',
              transform: 'translateX(0)', // Ensure no offset
            }}>
              <MotionBox
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                sx={{ 
                  width: '100%',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}
              >
                <Box sx={{
                  position: 'relative',
                  width: { xs: '280px', sm: '320px', md: '400px' },
                  height: { xs: '280px', sm: '320px', md: '400px' },
                  margin: '0 auto', // Center horizontally
                }}>
                  {/* Outer pulsing ring */}
                  <MotionBox
                    animate={controls}
                    sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '120%',
                      height: '120%',
                      borderRadius: '50%',
                      border: '2px solid rgba(255, 255, 255, 0.2)',
                      zIndex: 1
                    }}
                  />

                  {/* Main Circle */}
                  <Box sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '100%',
                    height: '100%',
                    borderRadius: '50%',
                    background: 'rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    overflow: 'hidden',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    zIndex: 2
                  }}>
                    <MotionBox
                      animate={{ 
                        rotate: 360
                      }}
                      transition={{ 
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                      sx={{
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        transformOrigin: 'center'
                      }}
                    >
                      <Description sx={{
                        fontSize: { xs: '80px', sm: '100px', md: '120px' },
                        color: 'white',
                        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))',
                        transform: 'translateX(0)', // Ensure no offset
                      }} />
                    </MotionBox>
                  </Box>

                  {/* Inner glowing circle */}
                  <MotionBox
                    animate={{ 
                      boxShadow: ['0 0 20px rgba(255,255,255,0.3)', '0 0 40px rgba(255,255,255,0.6)', '0 0 20px rgba(255,255,255,0.3)']
                    }}
                    transition={{ 
                      duration: 3,
                      repeat: Infinity,
                      repeatType: "reverse"
                    }}
                    sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '80%',
                      height: '80%',
                      borderRadius: '50%',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      zIndex: 1
                    }}
                  />

                  {/* Floating Icons */}
                  <Box sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '100%',
                    height: '100%',
                  }}>
                    <MotionBox
                      animate={{ 
                        rotate: 360
                      }}
                      transition={{ 
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                      sx={{
                        position: 'absolute',
                        width: '100%',
                        height: '100%',
                        transformOrigin: 'center',
                        zIndex: 3
                      }}
                    >
                      {features.map((feature, index) => (
                        <MotionBox
                          key={index}
                          whileHover={{ scale: 1.2 }}
                          sx={{
                            position: 'absolute',
                            top: `${50 + 45 * Math.sin(2 * Math.PI * index / features.length)}%`,
                            left: `${50 + 45 * Math.cos(2 * Math.PI * index / features.length)}%`,
                            transform: 'translate(-50%, -50%)',
                            background: 'rgba(255, 255, 255, 0.15)',
                            borderRadius: '50%',
                            p: { xs: 1, sm: 1.5, md: 2 },
                            backdropFilter: 'blur(5px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 2,
                          }}
                        >
                          {React.cloneElement(feature.icon, { 
                            sx: { 
                              fontSize: { xs: '24px', sm: '28px', md: '32px' },
                              color: 'white',
                              filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))'
                            } 
                          })}
                        </MotionBox>
                      ))}
                    </MotionBox>
                  </Box>
                </Box>
              </MotionBox>
            </Box>
          </Grid>
        </Grid>
      </Container>

      {/* Features Section */}
      <Box sx={{ 
        py: { xs: 6, md: 8 }, 
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        position: 'relative',
        zIndex: 1
      }}>
        <Container maxWidth="lg">
          <MotionBox
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <Typography 
              variant="h3" 
              align="center" 
              sx={{ 
                color: 'white',
                mb: { xs: 4, md: 6 },
                fontWeight: 'bold',
                fontSize: { xs: '2rem', sm: '2.5rem', md: '3rem' }
              }}
            >
              Why Choose DocuMorph AI
            </Typography>

            <Grid container spacing={3}>
              {features.map((feature, index) => (
                <Grid item xs={12} sm={6} md={3} key={index}>
                  <MotionBox variants={itemVariants}>
                    <Card sx={{ 
                      height: '100%',
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: 4,
                      transition: 'all 0.3s ease-in-out',
                      '&:hover': {
                        transform: 'translateY(-10px)',
                        backgroundColor: 'rgba(255, 255, 255, 0.08)',
                        boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                      },
                    }}>
                      <CardContent sx={{ 
                        p: 3,
                        textAlign: 'center',
                        color: 'white'
                      }}>
                        <MotionBox
                          whileHover={{ scale: 1.1, rotate: 5 }}
                          transition={{ type: "spring", stiffness: 400, damping: 10 }}
                          sx={{ mb: 2 }}
                        >
                          {React.cloneElement(feature.icon, { 
                            sx: { fontSize: '40px' } 
                          })}
                        </MotionBox>
                        <Typography 
                          variant="h6" 
                          gutterBottom 
                          sx={{ 
                            fontWeight: 'bold',
                            fontSize: { xs: '1.1rem', md: '1.25rem' }
                          }}
                        >
                          {feature.title}
                        </Typography>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            opacity: 0.9,
                            fontSize: { xs: '0.875rem', md: '1rem' }
                          }}
                        >
                          {feature.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </MotionBox>
                </Grid>
              ))}
            </Grid>
          </MotionBox>
        </Container>
      </Box>

      {/* Onboarding Dialog */}
      <OnboardingDialog 
        open={onboardingOpen}
        onClose={() => setOnboardingOpen(false)}
        onComplete={handleOnboardingComplete}
      />
    </Box>
  );
};

export default LandingPage;