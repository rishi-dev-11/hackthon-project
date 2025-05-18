import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Tab,
  Tabs,
  TextField,
  InputAdornment,
} from '@mui/material';
import { motion } from 'framer-motion';
import { Search as SearchIcon } from '@mui/icons-material';
import { useSubscription } from '../context/SubscriptionContext';
import FeatureGuard from '../components/FeatureGuard';
import TemplateCard from '../components/TemplateCard';

const TemplateSelection = () => {
  const [activeTab, setActiveTab] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const { userTier } = useSubscription();

  const templates = [
    {
      id: 1,
      title: 'Basic Academic Paper',
      description: 'Simple academic paper format with standard sections',
      category: 'academic',
      isPremium: false,
      thumbnailUrl: '/templates/academic-basic.png',
    },
    {
      id: 2,
      title: 'Advanced Research Paper',
      description: 'Professional research paper template with advanced formatting',
      category: 'academic',
      isPremium: true,
      thumbnailUrl: '/templates/academic-pro.png',
    },
    {
      id: 3,
      title: 'Business Report',
      description: 'Clean and professional business report template',
      category: 'business',
      isPremium: false,
      thumbnailUrl: '/templates/business-basic.png',
    },
    {
      id: 4,
      title: 'Corporate Branding Report',
      description: 'Premium business template with branding options',
      category: 'business',
      isPremium: true,
      thumbnailUrl: '/templates/business-pro.png',
    },
    {
      id: 5,
      title: 'Student Assignment',
      description: 'Basic template for student assignments',
      category: 'education',
      isPremium: false,
      thumbnailUrl: '/templates/student-basic.png',
    },
    {
      id: 6,
      title: 'Thesis Template',
      description: 'Advanced thesis template with automatic formatting',
      category: 'education',
      isPremium: true,
      thumbnailUrl: '/templates/thesis-pro.png',
    },
  ];

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = activeTab === 'all' || template.category === activeTab;
    const matchesAccess = userTier === 'premium' || !template.isPremium;
    
    return matchesSearch && matchesCategory && matchesAccess;
  });

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <motion
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Document Templates
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Choose from our collection of professional templates
          </Typography>
        </Box>

        <Box sx={{ mb: 4 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search templates..."
            value={searchQuery}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ mb: 3 }}
          />

          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="All Templates" value="all" />
            <Tab label="Academic" value="academic" />
            <Tab label="Business" value="business" />
            <Tab label="Education" value="education" />
          </Tabs>
        </Box>

        <Grid container spacing={3}>
          {filteredTemplates.map(template => (
            <Grid gridColumn={{ xs: 'span 12', sm: 'span 6', md: 'span 4' }} key={template.id}>
              {template.isPremium ? (
                <FeatureGuard feature="templates">
                  <TemplateCard
                    template={template}
                    isPremium={template.isPremium}
                  />
                </FeatureGuard>
              ) : (
                <TemplateCard
                  template={template}
                  isPremium={template.isPremium}
                />
              )}
            </Grid>
          ))}
        </Grid>

        {filteredTemplates.length === 0 && (
          <Box sx={{ 
            textAlign: 'center', 
            py: 8,
            color: 'text.secondary'
          }}>
            <Typography variant="h6">
              No templates found
            </Typography>
            <Typography variant="body2">
              Try adjusting your search or filters
            </Typography>
          </Box>
        )}
      </motion>
    </Container>
  );
};

export default TemplateSelection;