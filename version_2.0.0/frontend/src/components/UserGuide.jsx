import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  IconButton, 
  Collapse, 
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import { 
  Close as CloseIcon,
  HelpOutline as HelpIcon,
  CheckCircleOutline as CheckIcon,
  Info as InfoIcon,
  Lightbulb as TipIcon,
  KeyboardArrowDown as ExpandIcon,
  KeyboardArrowUp as CollapseIcon
} from '@mui/icons-material';
import { useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useSubscription } from '../context/SubscriptionContext';

const UserGuide = () => {
  const [expanded, setExpanded] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [hasSeenGuide, setHasSeenGuide] = useState(false);
  const location = useLocation();
  const { userTier } = useSubscription();
  const [guideContent, setGuideContent] = useState(null);

  // Check if user has seen the guide for this page
  useEffect(() => {
    const seenGuides = JSON.parse(localStorage.getItem('seenGuides') || '{}');
    const hasSeenCurrentGuide = seenGuides[location.pathname];
    setHasSeenGuide(hasSeenCurrentGuide);
    setDismissed(hasSeenCurrentGuide);
    setExpanded(false);
  }, [location.pathname]);

  // Set guide content based on current route
  useEffect(() => {
    const guides = {
      '/': {
        title: 'Welcome to DocuMorph AI',
        description: 'Transform your documents with AI-powered tools.',
        steps: [
          { icon: <InfoIcon color="info" />, text: 'Use our AI to enhance document formatting and style.' },
          { icon: <InfoIcon color="info" />, text: 'Upload documents and apply templates with a few clicks.' },
          { icon: <TipIcon color="warning" />, text: `You are currently on the ${userTier} plan. ${userTier === 'premium' ? 'You have access to all features.' : 'Upgrade to premium for additional features.'}` }
        ]
      },
      '/upload': {
        title: 'Document Upload',
        description: 'Start transforming your documents here.',
        steps: [
          { icon: <CheckIcon color="success" />, text: 'Drag and drop or click to select a document.' },
          { icon: <InfoIcon color="info" />, text: 'Supported formats: PDF, DOCX, TXT, and more.' },
          { icon: <TipIcon color="warning" />, text: 'Remember to check the document analysis before processing.' }
        ]
      },
      '/templates': {
        title: 'Document Templates',
        description: 'Choose from our collection of professional templates.',
        steps: [
          { icon: <CheckIcon color="success" />, text: 'Browse templates by category or search by name.' },
          { icon: <InfoIcon color="info" />, text: `Some templates require a premium subscription.` },
          { icon: <TipIcon color="warning" />, text: 'Click on a template to see a preview and apply it to your document.' }
        ]
      },
      '/dashboard': {
        title: 'Your Dashboard',
        description: 'Manage your documents and templates.',
        steps: [
          { icon: <CheckIcon color="success" />, text: 'View recent documents and quick actions.' },
          { icon: <InfoIcon color="info" />, text: 'Check your usage statistics and available features.' },
          { icon: <TipIcon color="warning" />, text: 'Use the sidebar menu to navigate to other sections.' }
        ]
      },
      '/profile': {
        title: 'Profile Settings',
        description: 'Manage your account information.',
        steps: [
          { icon: <CheckIcon color="success" />, text: 'Update your personal information and preferences.' },
          { icon: <InfoIcon color="info" />, text: 'Check your subscription details and history.' },
          { icon: <TipIcon color="warning" />, text: 'Enable two-factor authentication for added security.' }
        ]
      },
      '/upgrade': {
        title: 'Upgrade Your Plan',
        description: 'Get access to premium features.',
        steps: [
          { icon: <CheckIcon color="success" />, text: 'Compare available plans and their features.' },
          { icon: <InfoIcon color="info" />, text: 'Premium includes advanced templates and AI processing.' },
          { icon: <TipIcon color="warning" />, text: 'Annual plans offer significant savings compared to monthly billing.' }
        ]
      }
    };

    // Default guide for routes without specific content
    const defaultGuide = {
      title: 'DocuMorph AI',
      description: 'You are currently viewing: ' + location.pathname.slice(1),
      steps: [
        { icon: <InfoIcon color="info" />, text: 'Navigate using the sidebar menu.' },
        { icon: <TipIcon color="warning" />, text: 'Need help? Contact our support team.' }
      ]
    };

    setGuideContent(guides[location.pathname] || defaultGuide);
  }, [location.pathname, userTier]);

  const handleDismiss = () => {
    setDismissed(true);
    
    // Save that user has seen this guide
    const seenGuides = JSON.parse(localStorage.getItem('seenGuides') || '{}');
    seenGuides[location.pathname] = true;
    localStorage.setItem('seenGuides', JSON.stringify(seenGuides));
    setHasSeenGuide(true);
  };

  const handleReset = () => {
    // Clear the seen guide status for this page
    const seenGuides = JSON.parse(localStorage.getItem('seenGuides') || '{}');
    delete seenGuides[location.pathname];
    localStorage.setItem('seenGuides', JSON.stringify(seenGuides));
    setHasSeenGuide(false);
    setDismissed(false);
    setExpanded(true);
  };

  if (!guideContent || dismissed) {
    return (
      <Box sx={{ position: 'fixed', bottom: 20, right: 20, zIndex: 1000 }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <IconButton 
            color="primary" 
            size="large"
            onClick={() => hasSeenGuide ? handleReset() : setDismissed(false)}
            sx={{ 
              bgcolor: 'background.paper',
              boxShadow: 2,
              '&:hover': { bgcolor: 'background.paper', opacity: 0.9 }
            }}
          >
            <HelpIcon />
          </IconButton>
        </motion.div>
      </Box>
    );
  }

  return (
    <Box sx={{ position: 'fixed', bottom: 20, right: 20, zIndex: 1000, maxWidth: expanded ? 360 : 260 }}>
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Paper 
          elevation={3}
          sx={{ 
            p: 2, 
            borderRadius: 2,
            border: '1px solid',
            borderColor: 'divider'
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
              <HelpIcon sx={{ mr: 1, color: 'primary.main' }} />
              {guideContent.title}
            </Typography>
            
            <Box>
              <IconButton size="small" onClick={() => setExpanded(!expanded)}>
                {expanded ? <CollapseIcon /> : <ExpandIcon />}
              </IconButton>
              <IconButton size="small" onClick={handleDismiss}>
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>
          
          <Collapse in={expanded}>
            <Typography variant="body2" color="text.secondary" paragraph>
              {guideContent.description}
            </Typography>
            
            <Divider sx={{ my: 1 }} />
            
            <List dense disablePadding>
              {guideContent.steps.map((step, index) => (
                <ListItem key={index} disableGutters sx={{ py: 0.5 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    {step.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={step.text}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                </ListItem>
              ))}
            </List>
            
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
              <Button size="small" onClick={handleDismiss} variant="text">
                Got it
              </Button>
            </Box>
          </Collapse>
        </Paper>
      </motion.div>
    </Box>
  );
};

export default UserGuide; 