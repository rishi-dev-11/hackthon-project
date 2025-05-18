import React, { useEffect, useRef } from 'react';
import { 
  Container, 
  Typography, 
  Button, 
  Box, 
  Grid, 
  Card, 
  CardContent,
  Paper,
  Stack,
  Divider,
  Chip
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import gsap from 'gsap';
import {
  DocumentScanner,
  AutoFixHigh,
  PhotoLibrary,
  Translate,
  SmartToy,
  ContentPaste,
  UploadFile,
  Storage,
  Psychology,
  Groups
} from '@mui/icons-material';

// Feature data
const features = [
  {
    title: 'Smart Document Processing',
    description: 'AI-powered analysis and transformation of document structure and content.',
    icon: <DocumentScanner fontSize="large" color="primary" />
  },
  {
    title: 'Professional Templates',
    description: 'Choose from a variety of template styles for different document types and purposes.',
    icon: <AutoFixHigh fontSize="large" color="primary" />
  },
  {
    title: 'Table & Figure Recognition',
    description: 'Automatically detect and format tables and figures for better presentation.',
    icon: <PhotoLibrary fontSize="large" color="primary" />
  },
  {
    title: 'Multi-language Support',
    description: 'Process documents in multiple languages with intelligent OCR capabilities.',
    icon: <Translate fontSize="large" color="primary" />
  },
  {
    title: 'Style Enhancement',
    description: 'Improve document style, formatting, and readability with smart suggestions.',
    icon: <ContentPaste fontSize="large" color="primary" />
  },
  {
    title: 'AI-Powered Analysis',
    description: 'Leverage advanced AI to extract meaning and structure from your documents.',
    icon: <SmartToy fontSize="large" color="primary" />
  }
];

// Plan tier data
const tiers = [
  {
    name: 'Free',
    chipColor: '#4caf50',
    description: 'Get started with basic document processing and templates.',
    features: [
      'Up to 5 documents per month',
      'Basic document processing',
      'Access to essential templates',
      'Extract text and basic structure',
      'Save and download formatted documents'
    ]
  },
  {
    name: 'Premium',
    chipColor: '#0e6ba8',
    description: 'Enhanced features for professional document transformation.',
    features: [
      'Unlimited documents',
      'Advanced AI-powered processing',
      'Access to all premium templates',
      'Table and figure extraction',
      'Multi-language support',
      'Custom formatting rules',
      'Priority support'
    ]
  }
];

const LandingPage = () => {
  const navigate = useNavigate();
  
  // Refs for GSAP animations
  const heroRef = useRef(null);
  const featureRefs = useRef([]);
  const tierRefs = useRef([]);
  const ctaRef = useRef(null);
  
  // Set up animations
  useEffect(() => {
    // Hero animation
    gsap.fromTo(
      heroRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
    );
    
    // Feature cards animation
    featureRefs.current.forEach((ref, index) => {
      gsap.fromTo(
        ref,
        { opacity: 0, y: 20 },
        { 
          opacity: 1, 
          y: 0, 
          duration: 0.5, 
          delay: 0.1 * index,
          ease: "power2.out" 
        }
      );
    });
    
    // Tier cards animation
    tierRefs.current.forEach((ref, index) => {
      gsap.fromTo(
        ref,
        { opacity: 0, y: 30 },
        { 
          opacity: 1, 
          y: 0, 
          duration: 0.5, 
          delay: 0.3 + (0.1 * index),
          ease: "power2.out" 
        }
      );
    });
    
    // CTA section animation
    gsap.fromTo(
      ctaRef.current,
      { opacity: 0 },
      { opacity: 1, duration: 1, delay: 0.5, ease: "power1.out" }
    );
  }, []);

  return (
    <Container maxWidth="xl">
      {/* Hero section */}
      <Box 
        ref={heroRef}
        sx={{ 
          py: 8,
          mt: 4,
          textAlign: 'center',
          position: 'relative'
        }}
      >
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom
          sx={{
            fontWeight: 700,
            color: 'primary.dark'
          }}
        >
          DocuMorph AI
        </Typography>
        <Typography 
          variant="h5" 
          color="text.secondary" 
          paragraph 
          sx={{ 
            mb: 4,
            maxWidth: '800px',
            mx: 'auto'
          }}
        >
          Intelligent Document Transformation for Researchers, Students, and Professionals
        </Typography>
        <Stack 
          direction={{ xs: 'column', sm: 'row' }} 
          spacing={2} 
          justifyContent="center"
          sx={{ mb: 4 }}
        >
          <Button 
            variant="contained" 
            size="large" 
            color="primary"
            onClick={() => navigate('/upload')}
            startIcon={<UploadFile />}
            sx={{ 
              py: 1.5,
              px: 3,
              fontSize: '1.1rem'
            }}
          >
            Upload Document
          </Button>
          <Button 
            variant="outlined" 
            size="large"
            onClick={() => navigate('/templates')}
            startIcon={<AutoFixHigh />}
            sx={{ 
              py: 1.5,
              px: 3,
              fontSize: '1.1rem'
            }}
          >
            Browse Templates
          </Button>
        </Stack>
      </Box>

      {/* Features section */}
      <Box sx={{ py: 6 }}>
        <Typography 
          variant="h4" 
          component="h2" 
          gutterBottom 
          align="center" 
          sx={{ 
            mb: 6,
            color: 'primary.dark',
            fontWeight: 600
          }}
        >
          Transform Your Documents with AI
        </Typography>
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                ref={el => featureRefs.current[index] = el}
                sx={{ 
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  backgroundColor: index % 2 === 0 ? 'background.paper' : 'rgba(14, 107, 168, 0.05)'
                }}
              >
                <CardContent sx={{ 
                  display: 'flex', 
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                  p: 3,
                  height: '100%'
                }}>
                  <Box sx={{ mb: 2 }}>
                    {feature.icon}
                  </Box>
                  <Typography 
                    variant="h5" 
                    component="h3" 
                    gutterBottom
                    sx={{ fontWeight: 600, color: 'primary.dark' }}
                  >
                    {feature.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Tier comparison */}
      <Paper
        sx={{ 
          p: 4, 
          my: 6, 
          borderRadius: 4,
          background: 'linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%)'
        }}
      >
        <Typography 
          variant="h4" 
          gutterBottom 
          align="center"
          sx={{ 
            mb: 4,
            color: 'primary.dark',
            fontWeight: 600
          }}
        >
          Choose Your Plan
        </Typography>
        
        <Grid container spacing={4}>
          {tiers.map((tier, index) => (
            <Grid item xs={12} md={6} key={tier.name}>
              <Paper 
                ref={el => tierRefs.current[index] = el}
                elevation={3}
                sx={{ 
                  p: 4, 
                  height: '100%',
                  position: 'relative',
                  borderRadius: 3,
                  overflow: 'hidden',
                  boxShadow: tier.name === 'Premium' ? '0 8px 24px rgba(14, 107, 168, 0.15)' : '0 4px 12px rgba(0,0,0,0.08)'
                }}
              >
                <Chip 
                  label={tier.name} 
                  sx={{ 
                    position: 'absolute',
                    top: 16,
                    right: 16,
                    fontWeight: 600,
                    bgcolor: tier.chipColor,
                    color: 'white'
                  }} 
                />
                <Typography 
                  variant="h5" 
                  component="h3" 
                  gutterBottom
                  sx={{ fontWeight: 600, mb: 1, color: 'primary.dark' }}
                >
                  {tier.name} Plan
                </Typography>
                <Typography variant="body1" sx={{ mb: 3, color: 'text.secondary' }}>
                  {tier.description}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ mt: 2 }}>
                  {tier.features.map((feature, idx) => (
                    <Box key={idx} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box 
                        sx={{ 
                          width: 8, 
                          height: 8, 
                          borderRadius: '50%', 
                          bgcolor: 'primary.main',
                          mr: 2
                        }} 
                      />
                      <Typography variant="body1">{feature}</Typography>
                    </Box>
                  ))}
                </Box>
                <Box sx={{ mt: 4, textAlign: 'center' }}>
                  <Button 
                    variant={tier.name === 'Premium' ? 'contained' : 'outlined'}
                    color="primary"
                    size="large"
                    onClick={() => navigate(tier.name === 'Premium' ? '/upgrade' : '/register')}
                    sx={{ px: 4 }}
                  >
                    {tier.name === 'Premium' ? 'Upgrade Now' : 'Sign Up Free'}
                  </Button>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* CTA section */}
      <Box 
        ref={ctaRef}
        sx={{ 
          py: 8, 
          textAlign: 'center',
          mb: 4
        }}
      >
        <Typography
          variant="h4"
          component="h2"
          gutterBottom
          sx={{ fontWeight: 600, color: 'primary.dark' }}
        >
          Ready to transform your documents?
        </Typography>
        <Typography 
          variant="body1" 
          sx={{ 
            mb: 4, 
            maxWidth: '700px', 
            mx: 'auto',
            color: 'text.secondary'
          }}
        >
          Join thousands of researchers, students and professionals who use DocuMorph AI 
          to enhance their documents and save hours of formatting time.
        </Typography>
        <Stack 
          direction={{ xs: 'column', sm: 'row' }} 
          spacing={2} 
          justifyContent="center"
        >
          <Button 
            variant="contained" 
            size="large" 
            color="primary"
            onClick={() => navigate('/register')}
            sx={{ px: 4, py: 1.5 }}
          >
            Get Started for Free
          </Button>
          <Button 
            variant="outlined" 
            size="large"
            onClick={() => navigate('/about')}
            sx={{ px: 4, py: 1.5 }}
          >
            Learn More
          </Button>
        </Stack>
      </Box>
    </Container>
  );
};

export default LandingPage;