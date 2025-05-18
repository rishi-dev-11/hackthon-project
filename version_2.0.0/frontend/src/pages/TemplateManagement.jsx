import React, { useState, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  ContentCopy as DuplicateIcon,
  Save as SaveIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
  Close as CloseIcon,
  FormatPaint as FormatPaintIcon
} from '@mui/icons-material';
import { motion as Motion } from 'framer-motion';
import { pageTransition, staggerContainer, fadeInUp } from '../styles/animations';
import { templateTypes } from '../config/templateConfig';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TemplatePreview from '../components/TemplatePreview';
import DocumentFormatter from '../components/DocumentFormatter';
import FeatureGuard from '../components/FeatureGuard';

const TemplateManagement = () => {
  const [templates, setTemplates] = useState([
    {
      id: 1,
      name: "Academic Research Paper",
      description: "Format for academic research papers with sections for abstract, methodology, and references",
      sections: ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "References"],
      lastModified: "2024-03-15"
    },
    {
      id: 2,
      name: "Business Proposal",
      description: "Professional business proposal template with executive summary and financial projections",
      sections: ["Executive Summary", "Company Overview", "Market Analysis", "Financial Projections"],
      lastModified: "2024-03-14"
    },
    {
      id: 3,
      name: "Technical Documentation",
      description: "Technical documentation template with code examples and API references",
      sections: ["Overview", "Installation", "Usage", "API Reference", "Examples"],
      lastModified: "2024-03-13"
    }
  ]);

  const [openDialog, setOpenDialog] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState(null);
  const [newTemplate, setNewTemplate] = useState({
    name: '',
    description: '',
    sections: []
  });
  const [formatting, setFormatting] = useState(templateTypes.academic.formatting);
  const [previewMode, setPreviewMode] = useState(false);
  const fileInputRef = useRef(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [applyingTemplate, setApplyingTemplate] = useState(false);

  const handleOpenDialog = (template = null) => {
    if (template) {
      setEditingTemplate(template);
      setNewTemplate(template);
    } else {
      setEditingTemplate(null);
      setNewTemplate({ name: '', description: '', sections: [] });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingTemplate(null);
    setNewTemplate({ name: '', description: '', sections: [] });
  };

  const handleSaveTemplate = () => {
    if (editingTemplate) {
      setTemplates(templates.map(t =>
        t.id === editingTemplate.id ? { ...newTemplate, id: t.id } : t
      ));
    } else {
      setTemplates([...templates, { ...newTemplate, id: Date.now() }]);
    }
    handleCloseDialog();
  };

  const handleDeleteTemplate = (id) => {
    setTemplates(templates.filter(t => t.id !== id));
  };

  const handleDuplicateTemplate = (template) => {
    const duplicatedTemplate = {
      ...template,
      id: Date.now(),
      name: `${template.name} (Copy)`,
      lastModified: new Date().toISOString().split('T')[0]
    };
    setTemplates([...templates, duplicatedTemplate]);
  };

  const handleFormattingChange = (category, field, value) => {
    setFormatting(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value
      }
    }));
  };

  const validateTemplate = (template) => {
    const requiredFields = ['name', 'description', 'sections', 'formatting'];
    const missingFields = requiredFields.filter(field => !template[field]);
    
    if (missingFields.length > 0) {
      throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
    }

    // Validate formatting structure
    if (!template.formatting.fonts || !template.formatting.pageLayout) {
      throw new Error('Invalid formatting structure');
    }

    return true;
  };

  const handleImportTemplate = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        // Try to parse as JSON first
        let importedTemplate;
        try {
          importedTemplate = JSON.parse(e.target.result);
        } catch {
          // If not JSON, create template from file content
          importedTemplate = {
            name: file.name,
            description: `Imported from ${file.name}`,
            sections: ['Content'],
            formatting: {
              ...templateTypes.academic.formatting,
              fileType: file.type,
              originalContent: e.target.result
            }
          };
        }
        
        // Validate template structure
        validateTemplate(importedTemplate);

        // Check for duplicate template name
        const isDuplicate = templates.some(t => t.name === importedTemplate.name);
        if (isDuplicate) {
          setSnackbar({
            open: true,
            message: 'A template with this name already exists',
            severity: 'warning'
          });
          return;
        }

        // Add new template
        setTemplates(prev => [...prev, { 
          ...importedTemplate, 
          id: Date.now(),
          lastModified: new Date().toISOString().split('T')[0]
        }]);

        setSnackbar({
          open: true,
          message: 'File imported successfully',
          severity: 'success'
        });
      } catch (error) {
        console.error('Error importing file:', error);
        setSnackbar({
          open: true,
          message: `Error importing file: ${error.message}`,
          severity: 'error'
        });
      }
    };

    reader.onerror = () => {
      setSnackbar({
        open: true,
        message: 'Error reading file',
        severity: 'error'
      });
    };

    try {
      reader.readAsText(file);
    } catch (error) {
      console.error('Error reading file:', error);
      setSnackbar({
        open: true,
        message: 'Error reading file',
        severity: 'error'
      });
    }
  };

  const handleExportTemplate = (template) => {
    try {
      const templateData = {
        ...template,
        formatting,
        lastModified: new Date().toISOString()
      };
      
      const blob = new Blob([JSON.stringify(templateData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${template.name.toLowerCase().replace(/\s+/g, '-')}-template.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setSnackbar({
        open: true,
        message: 'Template exported successfully',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error exporting template:', error);
      setSnackbar({
        open: true,
        message: 'Error exporting template',
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleApplyTemplate = (template) => {
    setEditingTemplate(template);
    setSelectedDocument({
      title: template.name,
      sections: template.sections.map(section => ({
        title: section,
        content: `Content for ${section}`
      }))
    });
    setApplyingTemplate(true);
  };

  const handleSaveFormatting = (newFormatting) => {
    setTemplates(prev => prev.map(t => 
      t.id === editingTemplate.id 
        ? { ...t, formatting: newFormatting }
        : t
    ));
    setApplyingTemplate(false);
    setSnackbar({
      open: true,
      message: 'Template formatting saved successfully',
      severity: 'success'
    });
  };

  const renderEnhancedFormattingOptions = () => (
    <Box sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>Enhanced Formatting Rules</Typography>
      
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Advanced Page Layout</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Header Position</InputLabel>
                <Select
                  value={formatting.pageLayout.headerPosition || 'top'}
                  onChange={(e) => handleFormattingChange('pageLayout', 'headerPosition', e.target.value)}
                >
                  <MenuItem value="top">Top</MenuItem>
                  <MenuItem value="bottom">Bottom</MenuItem>
                  <MenuItem value="none">None</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Footer Position</InputLabel>
                <Select
                  value={formatting.pageLayout.footerPosition || 'bottom'}
                  onChange={(e) => handleFormattingChange('pageLayout', 'footerPosition', e.target.value)}
                >
                  <MenuItem value="top">Top</MenuItem>
                  <MenuItem value="bottom">Bottom</MenuItem>
                  <MenuItem value="none">None</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Advanced Typography</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {Object.entries(formatting.fonts).map(([key, value]) => (
              <Grid item xs={12} key={key}>
                <Typography variant="subtitle2" gutterBottom>
                  {key.charAt(0).toUpperCase() + key.slice(1)}
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <TextField
                      fullWidth
                      label="Font Family"
                      value={value.family}
                      onChange={(e) => handleFormattingChange('fonts', key, { ...value, family: e.target.value })}
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      fullWidth
                      label="Size"
                      value={value.size}
                      onChange={(e) => handleFormattingChange('fonts', key, { ...value, size: e.target.value })}
                    />
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      fullWidth
                      label="Weight"
                      select
                      value={value.weight}
                      onChange={(e) => handleFormattingChange('fonts', key, { ...value, weight: e.target.value })}
                    >
                      <MenuItem value="normal">Normal</MenuItem>
                      <MenuItem value="bold">Bold</MenuItem>
                      <MenuItem value="lighter">Lighter</MenuItem>
                    </TextField>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <TextField
                      fullWidth
                      label="Color"
                      type="color"
                      value={value.color}
                      onChange={(e) => handleFormattingChange('fonts', key, { ...value, color: e.target.value })}
                    />
                  </Grid>
                </Grid>
              </Grid>
            ))}
          </Grid>
        </AccordionDetails>
      </Accordion>

      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Advanced Spacing</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Line Height"
                type="number"
                value={formatting.spacing.lineHeight || 1.5}
                onChange={(e) => handleFormattingChange('spacing', 'lineHeight', parseFloat(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Paragraph Spacing"
                type="number"
                value={formatting.spacing.paragraph}
                onChange={(e) => handleFormattingChange('spacing', 'paragraph', parseFloat(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Section Spacing"
                type="number"
                value={formatting.spacing.section}
                onChange={(e) => handleFormattingChange('spacing', 'section', parseFloat(e.target.value))}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );

  const renderDialogContent = () => (
    <DialogContent>
      <Box sx={{ pt: 2 }}>
        <TextField
          fullWidth
          label="Template Name"
          value={newTemplate.name}
          onChange={(e) => setNewTemplate({ ...newTemplate, name: e.target.value })}
          margin="normal"
          required
        />
        <TextField
          fullWidth
          label="Description"
          value={newTemplate.description}
          onChange={(e) => setNewTemplate({ ...newTemplate, description: e.target.value })}
          margin="normal"
          multiline
          rows={3}
          required
        />
        <TextField
          fullWidth
          label="Sections (comma-separated)"
          value={newTemplate.sections.join(', ')}
          onChange={(e) => setNewTemplate({
            ...newTemplate,
            sections: e.target.value.split(',').map(s => s.trim()).filter(s => s)
          })}
          margin="normal"
          required
          helperText="Enter sections separated by commas"
        />
        {renderEnhancedFormattingOptions()}
      </Box>
    </DialogContent>
  );

  const renderTemplateCard = (template, index) => (
    <Grid item xs={12} md={6} lg={4} key={template.id}>
      <Motion.div
        variants={fadeInUp}
        transition={{ delay: index * 0.1 }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 3,
            height: '100%',
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: 4,
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            transition: 'all 0.3s ease',
            '&:hover': {
              transform: 'translateY(-5px)',
              boxShadow: '0 8px 30px rgba(0, 0, 0, 0.15)'
            }
          }}
        >
          <Typography variant="h5" gutterBottom color="primary">
            {template.name}
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            {template.description}
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            {template.sections.map((section, idx) => (
              <Chip
                key={idx}
                label={section}
                size="small"
                sx={{ mr: 1, mb: 1 }}
              />
            ))}
          </Box>

          <Typography variant="caption" color="text.secondary" display="block">
            Last modified: {template.lastModified}
          </Typography>

          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
            <Tooltip title="Preview">
              <IconButton onClick={() => {
                setEditingTemplate(template);
                setPreviewMode(true);
              }}>
                <VisibilityIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Edit">
              <IconButton onClick={() => handleOpenDialog(template)}>
                <EditIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Duplicate">
              <IconButton onClick={() => handleDuplicateTemplate(template)}>
                <DuplicateIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export">
              <IconButton onClick={() => handleExportTemplate(template)}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Apply Template">
              <IconButton onClick={() => handleApplyTemplate(template)}>
                <FormatPaintIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Delete">
              <IconButton onClick={() => handleDeleteTemplate(template.id)}>
                <DeleteIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Paper>
      </Motion.div>
    </Grid>
  );

  return (
    <Motion.div
      initial="initial"
      animate="animate"
      exit="exit"
      variants={pageTransition}
    >
      <Container maxWidth="lg">
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
              Template Management
            </Typography>
          </Motion.div>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mb: 4 }}>
            <input
              type="file"
              accept="*/*"
              style={{ display: 'none' }}
              ref={fileInputRef}
              onChange={handleImportTemplate}
            />
            <Motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant="outlined"
                startIcon={<UploadIcon />}
                onClick={() => fileInputRef.current.click()}
              >
                Import Template
              </Button>
            </Motion.div>
            <Motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => handleOpenDialog()}
                sx={{
                  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #1976D2 30%, #1CB5E0 90%)',
                  }
                }}
              >
                Create New Template
              </Button>
            </Motion.div>
          </Box>

          {previewMode ? (
            <Box sx={{ position: 'relative' }}>
              <IconButton
                sx={{ position: 'absolute', top: 0, right: 0, zIndex: 1 }}
                onClick={() => setPreviewMode(false)}
              >
                <CloseIcon />
              </IconButton>
              <TemplatePreview
                template={editingTemplate}
                formatting={formatting}
              />
            </Box>
          ) : (
            <Motion.div variants={staggerContainer} initial="initial" animate="animate">
              <Grid container spacing={4}>
                {templates.map((template, index) => renderTemplateCard(template, index))}
              </Grid>
            </Motion.div>
          )}
        </Box>

        <Dialog
          open={openDialog}
          onClose={handleCloseDialog}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            {editingTemplate ? 'Edit Template' : 'Create New Template'}
          </DialogTitle>
          {renderDialogContent()}
          <DialogActions>
            <Button onClick={handleCloseDialog}>Cancel</Button>
            <Button
              onClick={handleSaveTemplate}
              variant="contained"
              startIcon={<SaveIcon />}
            >
              Save Template
            </Button>
          </DialogActions>
        </Dialog>

        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert 
            onClose={handleCloseSnackbar} 
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>

        {applyingTemplate && (
          <DocumentFormatter
            document={selectedDocument}
            template={editingTemplate}
            onSave={handleSaveFormatting}
          />
        )}
      </Container>
    </Motion.div>
  );
};

export default TemplateManagement;