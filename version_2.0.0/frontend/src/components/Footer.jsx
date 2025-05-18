import React from 'react';
import { Box, Container, Grid, Typography, Link, Divider, Stack, IconButton } from '@mui/material';
import { 
  GitHub, 
  LinkedIn, 
  Twitter, 
  Facebook,
  Email
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const Footer = () => {
  const navigate = useNavigate();
  const currentYear = new Date().getFullYear();

  const footerLinks = [
    {
      title: 'Product',
      links: [
        { name: 'Features', path: '/features' },
        { name: 'Pricing', path: '/upgrade' },
        { name: 'Tutorials', path: '/tutorials' },
        { name: 'Templates', path: '/templates' }
      ]
    },
    {
      title: 'Resources',
      links: [
        { name: 'Documentation', path: '/docs' },
        { name: 'API', path: '/api' },
        { name: 'Support', path: '/support' },
        { name: 'FAQ', path: '/faq' }
      ]
    },
    {
      title: 'Company',
      links: [
        { name: 'About', path: '/about' },
        { name: 'Contact', path: '/contact' },
        { name: 'Terms', path: '/terms' },
        { name: 'Privacy', path: '/privacy' }
      ]
    }
  ];

  const socialLinks = [
    { icon: <GitHub />, url: 'https://github.com' },
    { icon: <LinkedIn />, url: 'https://linkedin.com' },
    { icon: <Twitter />, url: 'https://twitter.com' },
    { icon: <Facebook />, url: 'https://facebook.com' },
    { icon: <Email />, url: 'mailto:contact@documorph.ai' }
  ];

  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 6,
        mt: 'auto',
        bgcolor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider'
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid gridColumn={{ xs: 'span 12', md: 'span 4' }}>
            <Box sx={{ mb: 3 }}>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 700, 
                  color: 'primary.main',
                  mb: 1
                }}
              >
                DocuMorph AI
              </Typography>
              <Typography 
                variant="body2" 
                color="text.secondary"
                sx={{ mb: 2 }}
              >
                Intelligent document transformation powered by AI. Transform your documents with smart formatting, style enhancements, and automated layout.
              </Typography>
            </Box>

            <Stack 
              direction="row" 
              spacing={1}
              sx={{ mb: 3 }}
            >
              {socialLinks.map((social, index) => (
                <IconButton 
                  key={index}
                  component="a"
                  href={social.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  size="small"
                  sx={{ 
                    color: 'text.secondary',
                    '&:hover': {
                      color: 'primary.main'
                    }
                  }}
                >
                  {social.icon}
                </IconButton>
              ))}
            </Stack>
          </Grid>

          {footerLinks.map((section) => (
            <Grid gridColumn={{ xs: 'span 12', sm: 'span 4', md: 'span 2.5' }} key={section.title}>
              <Typography 
                variant="subtitle2" 
                color="text.primary"
                sx={{ 
                  fontWeight: 600,
                  mb: 2
                }}
              >
                {section.title}
              </Typography>
              <Stack spacing={1.5}>
                {section.links.map((link) => (
                  <Link
                    key={link.name}
                    component="button"
                    variant="body2"
                    color="text.secondary"
                    underline="hover"
                    onClick={() => navigate(link.path)}
                    sx={{ 
                      textAlign: 'left',
                      '&:hover': {
                        color: 'primary.main'
                      }
                    }}
                  >
                    {link.name}
                  </Link>
                ))}
              </Stack>
            </Grid>
          ))}
        </Grid>

        <Divider sx={{ my: 4 }} />

        <Box 
          sx={{ 
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: { xs: 'center', sm: 'center' },
            gap: 2
          }}
        >
          <Typography 
            variant="body2" 
            color="text.secondary"
          >
            &copy; {currentYear} DocuMorph AI. All rights reserved.
          </Typography>
          
          <Stack 
            direction="row" 
            spacing={3}
            sx={{
              display: 'flex',
              justifyContent: 'center'
            }}
          >
            <Link 
              component="button"
              variant="body2" 
              color="text.secondary"
              underline="hover"
              onClick={() => navigate('/terms')}
            >
              Terms of Service
            </Link>
            <Link 
              component="button"
              variant="body2" 
              color="text.secondary"
              underline="hover"
              onClick={() => navigate('/privacy')}
            >
              Privacy Policy
            </Link>
            <Link 
              component="button"
              variant="body2" 
              color="text.secondary"
              underline="hover"
              onClick={() => navigate('/cookies')}
            >
              Cookies
            </Link>
          </Stack>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
