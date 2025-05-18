import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  IconButton,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip
} from '@mui/material';
import {
  FormatBold,
  FormatItalic,
  FormatUnderlined,
  FormatAlignLeft,
  FormatAlignCenter,
  FormatAlignRight,
  ZoomIn,
  ZoomOut,
  Save as SaveIcon,
  Preview as PreviewIcon
} from '@mui/icons-material';
import { motion as Motion } from 'framer-motion';
import { fadeInUp } from '../styles/animations';
import FeatureGuard from './FeatureGuard';

const AdvancedFormatting = () => (
  <Box>
    {/* Advanced formatting options here */}
  </Box>
);

const DocumentFormatter = ({ document, template, onSave }) => {
  const [formatting, setFormatting] = useState(template.formatting);
  const [previewMode, setPreviewMode] = useState(false);
  const [zoom, setZoom] = useState(100);

  const handleFormattingChange = (category, field, value) => {
    setFormatting(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value
      }
    }));
  };

  const renderFormattingToolbar = () => (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        mb: 2,
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(10px)',
        borderRadius: 2
      }}
    >
      <Grid container spacing={2} alignItems="center">
        <Grid item>
          <Tooltip title="Bold">
            <IconButton
              onClick={() => handleFormattingChange('fonts', 'body', {
                ...formatting.fonts.body,
                weight: formatting.fonts.body.weight === 'bold' ? 'normal' : 'bold'
              })}
            >
              <FormatBold />
            </IconButton>
          </Tooltip>
        </Grid>
        <Grid item>
          <Tooltip title="Italic">
            <IconButton
              onClick={() => handleFormattingChange('fonts', 'body', {
                ...formatting.fonts.body,
                style: formatting.fonts.body.style === 'italic' ? 'normal' : 'italic'
              })}
            >
              <FormatItalic />
            </IconButton>
          </Tooltip>
        </Grid>
        <Grid item>
          <Tooltip title="Underline">
            <IconButton
              onClick={() => handleFormattingChange('fonts', 'body', {
                ...formatting.fonts.body,
                decoration: formatting.fonts.body.decoration === 'underline' ? 'none' : 'underline'
              })}
            >
              <FormatUnderlined />
            </IconButton>
          </Tooltip>
        </Grid>
        <Grid item>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Font Size</InputLabel>
            <Select
              value={formatting.fonts.body.size}
              onChange={(e) => handleFormattingChange('fonts', 'body', {
                ...formatting.fonts.body,
                size: e.target.value
              })}
            >
              <MenuItem value="12pt">12pt</MenuItem>
              <MenuItem value="14pt">14pt</MenuItem>
              <MenuItem value="16pt">16pt</MenuItem>
              <MenuItem value="18pt">18pt</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item>
          <Tooltip title="Zoom In">
            <IconButton onClick={() => setZoom(prev => Math.min(prev + 10, 200))}>
              <ZoomIn />
            </IconButton>
          </Tooltip>
        </Grid>
        <Grid item>
          <Tooltip title="Zoom Out">
            <IconButton onClick={() => setZoom(prev => Math.max(prev - 10, 50))}>
              <ZoomOut />
            </IconButton>
          </Tooltip>
        </Grid>
      </Grid>
    </Paper>
  );

  const renderPreview = () => (
    <Paper
      elevation={0}
      sx={{
        p: 4,
        background: 'white',
        borderRadius: 2,
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
        transform: `scale(${zoom / 100})`,
        transformOrigin: 'top left',
        transition: 'transform 0.3s ease'
      }}
    >
      <Typography
        variant="h4"
        sx={{
          fontFamily: formatting.fonts.title.family,
          fontSize: formatting.fonts.title.size,
          fontWeight: formatting.fonts.title.weight,
          color: formatting.fonts.title.color,
          textAlign: 'center',
          mb: 4
        }}
      >
        {document.title}
      </Typography>

      {document.sections.map((section, index) => (
        <Box key={index} sx={{ mb: 4 }}>
          <Typography
            variant="h5"
            sx={{
              fontFamily: formatting.fonts.heading1.family,
              fontSize: formatting.fonts.heading1.size,
              fontWeight: formatting.fonts.heading1.weight,
              color: formatting.fonts.heading1.color,
              mb: 2
            }}
          >
            {section.title}
          </Typography>
          <Typography
            sx={{
              fontFamily: formatting.fonts.body.family,
              fontSize: formatting.fonts.body.size,
              fontWeight: formatting.fonts.body.weight,
              color: formatting.fonts.body.color,
              lineHeight: formatting.spacing.lineHeight
            }}
          >
            {section.content}
          </Typography>
        </Box>
      ))}
    </Paper>
  );

  return (
    <Motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={fadeInUp}
    >
      <Box sx={{ p: 3 }}>
        {renderFormattingToolbar()}
        
        <Box sx={{ 
          position: 'relative',
          overflow: 'auto',
          maxHeight: 'calc(100vh - 200px)',
          p: 2
        }}>
          {renderPreview()}
        </Box>

        <Box sx={{ 
          position: 'fixed',
          bottom: 20,
          right: 20,
          display: 'flex',
          gap: 2
        }}>
          <Button
            variant="outlined"
            startIcon={<PreviewIcon />}
            onClick={() => setPreviewMode(!previewMode)}
          >
            {previewMode ? 'Edit' : 'Preview'}
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={() => onSave(formatting)}
            sx={{
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
              '&:hover': {
                background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
              }
            }}
          >
            Save Formatting
          </Button>
        </Box>

        {/* Advanced formatting options only for premium users */}
        <FeatureGuard feature="advanced-formatting">
          <AdvancedFormatting />
        </FeatureGuard>
      </Box>
    </Motion.div>
  );
};

export default DocumentFormatter;