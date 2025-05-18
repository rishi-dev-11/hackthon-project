import React from 'react';
import { Paper, Box, Typography, Grid } from '@mui/material';
import { motion as Motion } from 'framer-motion';
import { fadeInUp } from '../styles/animations';

const TemplatePreview = ({ template, formatting }) => {
  const renderPreviewContent = () => {
    switch (template.type) {
      case 'academic':
        return (
          <>
            <Typography variant="h4" sx={{ 
              fontFamily: formatting.fonts.title.family,
              fontSize: formatting.fonts.title.size,
              fontWeight: formatting.fonts.title.weight,
              color: formatting.fonts.title.color,
              textAlign: 'center',
              mb: 4
            }}>
              {template.name}
            </Typography>
            
            <Typography variant="h5" sx={{
              fontFamily: formatting.fonts.heading1.family,
              fontSize: formatting.fonts.heading1.size,
              fontWeight: formatting.fonts.heading1.weight,
              color: formatting.fonts.heading1.color,
              mb: 2
            }}>
              Abstract
            </Typography>
            
            <Typography sx={{
              fontFamily: formatting.fonts.body.family,
              fontSize: formatting.fonts.body.size,
              fontWeight: formatting.fonts.body.weight,
              color: formatting.fonts.body.color,
              mb: 4
            }}>
              This is a sample abstract text that demonstrates the formatting of body text in the academic template.
            </Typography>

            <Typography variant="h5" sx={{
              fontFamily: formatting.fonts.heading1.family,
              fontSize: formatting.fonts.heading1.size,
              fontWeight: formatting.fonts.heading1.weight,
              color: formatting.fonts.heading1.color,
              mb: 2
            }}>
              Introduction
            </Typography>

            <Typography sx={{
              fontFamily: formatting.fonts.body.family,
              fontSize: formatting.fonts.body.size,
              fontWeight: formatting.fonts.body.weight,
              color: formatting.fonts.body.color,
              mb: 2
            }}>
              This is a sample introduction paragraph that demonstrates the formatting of body text in the academic template.
            </Typography>
          </>
        );
      case 'business':
        return (
          <>
            <Typography variant="h4" sx={{ 
              fontFamily: formatting.fonts.title.family,
              fontSize: formatting.fonts.title.size,
              fontWeight: formatting.fonts.title.weight,
              color: formatting.fonts.title.color,
              textAlign: 'center',
              mb: 4
            }}>
              {template.name}
            </Typography>
            
            <Typography variant="h5" sx={{
              fontFamily: formatting.fonts.heading1.family,
              fontSize: formatting.fonts.heading1.size,
              fontWeight: formatting.fonts.heading1.weight,
              color: formatting.fonts.heading1.color,
              mb: 2
            }}>
              Executive Summary
            </Typography>
            
            <Typography sx={{
              fontFamily: formatting.fonts.body.family,
              fontSize: formatting.fonts.body.size,
              fontWeight: formatting.fonts.body.weight,
              color: formatting.fonts.body.color,
              mb: 4
            }}>
              This is a sample executive summary that demonstrates the formatting of body text in the business proposal template.
            </Typography>
          </>
        );
      case 'technical':
        return (
          <>
            <Typography variant="h4" sx={{ 
              fontFamily: formatting.fonts.title.family,
              fontSize: formatting.fonts.title.size,
              fontWeight: formatting.fonts.title.weight,
              color: formatting.fonts.title.color,
              textAlign: 'center',
              mb: 4
            }}>
              {template.name}
            </Typography>
            
            <Typography variant="h5" sx={{
              fontFamily: formatting.fonts.heading1.family,
              fontSize: formatting.fonts.heading1.size,
              fontWeight: formatting.fonts.heading1.weight,
              color: formatting.fonts.heading1.color,
              mb: 2
            }}>
              Overview
            </Typography>
            
            <Typography sx={{
              fontFamily: formatting.fonts.body.family,
              fontSize: formatting.fonts.body.size,
              fontWeight: formatting.fonts.body.weight,
              color: formatting.fonts.body.color,
              mb: 4
            }}>
              This is a sample overview that demonstrates the formatting of body text in the technical documentation template.
            </Typography>

            <Box sx={{ 
              p: 2, 
              bgcolor: '#f5f5f5', 
              borderRadius: 1,
              fontFamily: formatting.fonts.code.family,
              fontSize: formatting.fonts.code.size,
              color: formatting.fonts.code.color
            }}>
              <pre>// Sample code block</pre>
            </Box>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <Motion.div variants={fadeInUp}>
      <Paper
        elevation={0}
        sx={{
          p: 4,
          background: 'white',
          borderRadius: 2,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          width: '100%',
          height: '100%',
          overflow: 'auto'
        }}
      >
        {renderPreviewContent()}
      </Paper>
    </Motion.div>
  );
};

export default TemplatePreview; 