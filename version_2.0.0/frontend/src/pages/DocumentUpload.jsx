import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  Grid,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stepper,
  Step,
  StepLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  TextField,
  CircularProgress,
  Alert,
  Stack,
  IconButton
} from '@mui/material';
import { 
  Upload as UploadIcon, 
  Check, 
  Description, 
  Image, 
  FormatListBulleted,
  Spellcheck, 
  Translate, 
  AutoFixHigh,
  FilterAlt,
  DeleteOutline
} from '@mui/icons-material';
import FeatureGuard from '../components/FeatureGuard';
import FeatureLimit from '../components/FeatureLimit';
import { useUser } from '../hooks/useUser';
import { useSubscription } from '../hooks/useSubscription';

const ACCEPTED_TYPES = {
  '.pdf': 'PDF Document',
  '.doc': 'Word Document (Legacy)',
  '.docx': 'Word Document',
  '.txt': 'Text File',
  '.rtf': 'Rich Text Format',
  '.odt': 'OpenDocument Text'
};

const STEPS = ['Upload', 'Configure', 'Process', 'Download'];

const DocumentUpload = () => {
  const [file, setFile] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [languageDetected, setLanguageDetected] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('');
  const [extractedElements, setExtractedElements] = useState({
    tables: 0,
    figures: 0,
    sections: 0
  });
  const [enhancementOptions, setEnhancementOptions] = useState({
    styleCorrection: true,
    smartSuggestions: true,
    tableFormatting: true,
    figureCaptioning: true,
    plagiarismCheck: false
  });
  
  const { userTier } = useSubscription();
  const isPremium = userTier === 'premium';

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setActiveStep(1);
      
      // Simulate detected language
      setTimeout(() => {
        setLanguageDetected('English');
      }, 500);
      
      // Simulate extracted elements
      setTimeout(() => {
        setExtractedElements({
          tables: Math.floor(Math.random() * 5),
          figures: Math.floor(Math.random() * 4),
          sections: Math.floor(Math.random() * 10) + 3
        });
      }, 1000);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setActiveStep(0);
    setLanguageDetected('');
    setExtractedElements({
      tables: 0,
      figures: 0,
      sections: 0
    });
    setProcessingComplete(false);
  };

  const handleProcessing = () => {
    setProcessing(true);
    setActiveStep(2);
    
    // Simulate processing
    setTimeout(() => {
      setProcessing(false);
      setProcessingComplete(true);
      setActiveStep(3);
    }, 3000);
  };

  const renderFileUploadArea = () => (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 4,
        borderRadius: 2,
        border: '2px dashed',
        borderColor: 'divider',
        bgcolor: 'background.paper',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        '&:hover': {
          borderColor: 'primary.main',
          bgcolor: 'rgba(14, 107, 168, 0.05)',
        }
      }}
    >
      <input
        type="file"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        id="document-upload"
        accept={Object.keys(ACCEPTED_TYPES).join(',')}
      />
      <label htmlFor="document-upload">
        <Box sx={{ py: 8 }}>
          <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Drag and drop your document here
          </Typography>
          <Typography variant="body2" color="text.secondary">
            or click to browse
          </Typography>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Accepted file types:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 1, mt: 1 }}>
              {Object.entries(ACCEPTED_TYPES).map(([ext, name]) => (
                <Chip 
                  key={ext} 
                  label={`${ext} (${name})`} 
                  size="small" 
                  variant="outlined" 
                  sx={{ m: 0.5 }} 
                />
              ))}
            </Box>
          </Box>
        </Box>
      </label>
    </Paper>
  );

  const renderFileDetails = () => (
    <Paper 
      elevation={1} 
      sx={{ 
        p: 3, 
        borderRadius: 2,
        position: 'relative'
      }}
    >
      <IconButton 
        size="small" 
        sx={{ position: 'absolute', top: 8, right: 8 }}
        onClick={handleRemoveFile}
      >
        <DeleteOutline />
      </IconButton>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Description sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Box>
          <Typography variant="h6">{file?.name}</Typography>
          <Typography variant="body2" color="text.secondary">
            {(file?.size / 1024).toFixed(2)} KB • {file?.type || 'Unknown type'}
          </Typography>
        </Box>
      </Box>
      
      <Divider sx={{ my: 2 }} />
      
      <Typography variant="subtitle1" gutterBottom>
        Document Analysis
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 4' }}>
          <Box sx={{ textAlign: 'center', p: 1 }}>
            <Typography variant="body2" color="text.secondary">Language</Typography>
            <Typography variant="h6">
              {languageDetected || (
                <CircularProgress size={16} sx={{ mt: 1 }} />
              )}
            </Typography>
          </Box>
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 4' }}>
          <Box sx={{ textAlign: 'center', p: 1 }}>
            <Typography variant="body2" color="text.secondary">Tables</Typography>
            <Typography variant="h6">
              {extractedElements.tables > 0 ? extractedElements.tables : (
                <CircularProgress size={16} sx={{ mt: 1 }} />
              )}
            </Typography>
          </Box>
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 4' }}>
          <Box sx={{ textAlign: 'center', p: 1 }}>
            <Typography variant="body2" color="text.secondary">Figures</Typography>
            <Typography variant="h6">
              {extractedElements.figures > 0 ? extractedElements.figures : (
                <CircularProgress size={16} sx={{ mt: 1 }} />
              )}
            </Typography>
          </Box>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Processing Options
        </Typography>
        
        <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 2 }}>
          <InputLabel>Target Language (Translation)</InputLabel>
          <Select
            value={targetLanguage}
            label="Target Language (Translation)"
            onChange={(e) => setTargetLanguage(e.target.value)}
            disabled={!isPremium}
          >
            <MenuItem value=""><em>No translation</em></MenuItem>
            <MenuItem value="en">English</MenuItem>
            <MenuItem value="fr">French</MenuItem>
            <MenuItem value="es">Spanish</MenuItem>
            <MenuItem value="de">German</MenuItem>
            <MenuItem value="hi">Hindi</MenuItem>
            {!isPremium && (
              <MenuItem disabled value="premium">
                <Chip size="small" label="Premium" color="secondary" /> More languages
              </MenuItem>
            )}
          </Select>
        </FormControl>
        
        <Grid container spacing={2}>
          {Object.entries(enhancementOptions).map(([key, value]) => (
            <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }} key={key}>
              <FormControl fullWidth variant="outlined" size="small">
                <TextField
                  select
                  label={key.replace(/([A-Z])/g, ' $1').trim()}
                  value={value ? "enabled" : "disabled"}
                  onChange={(e) => setEnhancementOptions({
                    ...enhancementOptions, 
                    [key]: e.target.value === "enabled"
                  })}
                  size="small"
                  disabled={key === 'plagiarismCheck' && !isPremium}
                >
                  <MenuItem value="enabled">Enabled</MenuItem>
                  <MenuItem value="disabled">Disabled</MenuItem>
                </TextField>
              </FormControl>
            </Grid>
          ))}
        </Grid>
        
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleProcessing}
            disabled={processing}
            startIcon={processing ? <CircularProgress size={20} /> : <AutoFixHigh />}
            sx={{
              py: 1.5,
              px: 3
            }}
          >
            {processing ? 'Processing...' : 'Process Document'}
          </Button>
        </Box>
      </Box>
    </Paper>
  );

  const renderProcessing = () => (
    <Paper elevation={1} sx={{ p: 4, textAlign: 'center', borderRadius: 2 }}>
      <Typography variant="h6" gutterBottom>
        Processing your document...
      </Typography>
      
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 4 }}>
        <CircularProgress size={60} sx={{ mb: 3 }} />
        <Typography variant="body1" sx={{ mb: 1 }}>
          Analyzing document structure...
        </Typography>
        <Typography variant="body2" color="text.secondary">
          This may take a few moments
        </Typography>
      </Box>
      
      <List sx={{ width: '100%', maxWidth: 500, mx: 'auto', textAlign: 'left' }}>
        <ListItem>
          <ListItemIcon><Check color="success" /></ListItemIcon>
          <ListItemText 
            primary="Document loaded successfully" 
            secondary={file?.name} 
          />
        </ListItem>
        <ListItem>
          <ListItemIcon><Check color="success" /></ListItemIcon>
          <ListItemText 
            primary="Content extraction complete" 
          />
        </ListItem>
        <ListItem>
          <ListItemIcon>
            {processing ? <CircularProgress size={20} /> : <Check color="success" />}
          </ListItemIcon>
          <ListItemText 
            primary="AI enhancements in progress" 
            secondary={processing ? "Applying style corrections and formatting..." : "Complete"}
          />
        </ListItem>
      </List>
    </Paper>
  );
  
  const renderDownload = () => (
    <Paper elevation={1} sx={{ p: 4, borderRadius: 2 }}>
      <Alert severity="success" sx={{ mb: 3 }}>
        Your document has been successfully processed!
      </Alert>
      
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Document Ready
        </Typography>
        <Typography variant="body1" paragraph>
          Your enhanced document is now ready for download or further actions.
        </Typography>
      </Box>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid gridColumn={{ xs: 'span 12', md: 'span 6' }}>
          <Button
            variant="contained"
            fullWidth
            sx={{ py: 2 }}
            onClick={() => {}}
          >
            Download Document
          </Button>
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', md: 'span 6' }}>
          <Button
            variant="outlined"
            fullWidth
            sx={{ py: 2 }}
            onClick={() => {}}
          >
            Preview Document
          </Button>
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="subtitle1" gutterBottom>
        Additional Options
      </Typography>
      
      <Grid container spacing={2}>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <Button
            variant="outlined"
            fullWidth
            startIcon={<FormatListBulleted />}
            onClick={() => {}}
            sx={{ mb: 2 }}
          >
            Apply Template
          </Button>
        </Grid>
        <Grid gridColumn={{ xs: 'span 12', sm: 'span 6' }}>
          <Button
            variant="outlined"
            fullWidth
            startIcon={<Translate />}
            onClick={() => {}}
            sx={{ mb: 2 }}
            disabled={!isPremium}
          >
            Translate Document
          </Button>
        </Grid>
        {isPremium && (
          <Grid gridColumn={{ xs: 'span 12' }}>
            <Button
              variant="outlined"
              fullWidth
              color="secondary"
              onClick={() => {}}
              sx={{ mb: 2 }}
            >
              Export to Google Drive
            </Button>
          </Grid>
        )}
      </Grid>
    </Paper>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box
        sx={{
          opacity: 1,
          transform: 'translateY(0)',
          transition: 'opacity 0.5s, transform 0.5s'
        }}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, color: 'primary.dark' }}>
          Document Transformation
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Upload your document and let our AI transform it according to your needs
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <Stepper activeStep={activeStep} alternativeLabel>
            {STEPS.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </Box>

        <Grid container spacing={4}>
          <Grid gridColumn={{ xs: 'span 12', md: 'span 8' }}>
            {activeStep === 0 && renderFileUploadArea()}
            {activeStep === 1 && file && renderFileDetails()}
            {activeStep === 2 && processing && renderProcessing()}
            {activeStep === 3 && processingComplete && renderDownload()}
          </Grid>

          <Grid gridColumn={{ xs: 'span 12', md: 'span 4' }}>
            <Paper elevation={0} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'primary.dark' }}>
                Document AI Features
              </Typography>
              <Divider sx={{ my: 2 }} />
              
              <List disablePadding>
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <FormatListBulleted color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Smart Document Structure Analysis"
                    secondary="AI identifies headers, sections and components"
                  />
                </ListItem>
                
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Image color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Table & Figure Recognition" 
                    secondary="Auto-number and format detected elements"
                  />
                </ListItem>
                
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Spellcheck color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Style Enhancement"
                    secondary="Improve readability and formatting"
                  />
                </ListItem>
                
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <AutoFixHigh color={isPremium ? "primary" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography>Advanced AI Suggestions</Typography>
                        {!isPremium && <Chip size="small" label="Premium" color="secondary" />}
                      </Stack>
                    }
                    secondary="Get content improvement recommendations"
                    primaryTypographyProps={{
                      color: isPremium ? 'text.primary' : 'text.disabled'
                    }}
                    secondaryTypographyProps={{
                      color: isPremium ? 'text.secondary' : 'text.disabled'
                    }}
                  />
                </ListItem>
                
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <Translate color={isPremium ? "primary" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography>Multi-language Support</Typography>
                        {!isPremium && <Chip size="small" label="Premium" color="secondary" />}
                      </Stack>
                    }
                    secondary="Process documents in 10+ languages"
                    primaryTypographyProps={{
                      color: isPremium ? 'text.primary' : 'text.disabled'
                    }}
                    secondaryTypographyProps={{
                      color: isPremium ? 'text.secondary' : 'text.disabled'
                    }}
                  />
                </ListItem>
              </List>
              
              {!isPremium && (
                <Button
                  variant="contained"
                  color="secondary"
                  fullWidth
                  sx={{ mt: 3 }}
                  onClick={() => {}}
                >
                  Upgrade to Premium
                </Button>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default DocumentUpload;