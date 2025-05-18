import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  Grid,
  Divider,
} from '@mui/material';
import { motion as Motion } from 'framer-motion';
import { Upload as UploadIcon } from '@mui/icons-material';
import FeatureGuard from '../components/FeatureGuard';
import FeatureLimit from '../components/FeatureLimit';
import { useUser } from '../hooks/useUser';

const DocumentUpload = () => {
  const [file, setFile] = useState(null);
  const { getUserSubscription } = useUser();
  const currentSubscription = getUserSubscription();

  const handleFileSelect = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = () => {
    // Handle file upload logic
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom>
          Upload Document
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" paragraph>
          Upload your document and let our AI transform it according to your needs
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12} md={8}>
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
                  bgcolor: 'rgba(33, 150, 243, 0.05)',
                }
              }}
            >
              <input
                type="file"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
                id="document-upload"
                accept=".doc,.docx,.pdf,.txt"
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
                  {file && (
                    <Typography variant="body2" color="primary" sx={{ mt: 2 }}>
                      Selected: {file.name}
                    </Typography>
                  )}
                </Box>
              </label>
            </Paper>

            {file && (
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleUpload}
                  sx={{
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                  }}
                >
                  Process Document
                </Button>
              </Box>
            )}
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper elevation={0} sx={{ p: 3, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom>
                Advanced Features
              </Typography>
              <Divider sx={{ my: 2 }} />
              
              <FeatureGuard feature="ocr">
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    OCR Processing
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Extract text from scanned documents and images
                  </Typography>
                </Box>
              </FeatureGuard>

              <FeatureGuard feature="language">
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Language Detection
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Automatic language detection and processing
                  </Typography>
                </Box>
              </FeatureGuard>

              {currentSubscription === 'free' && (
                <FeatureLimit 
                  featureKey="templates"
                  message="Upgrade to access premium templates and advanced formatting options"
                />
              )}
            </Paper>
          </Grid>
        </Grid>
      </Motion.div>
    </Container>
  );
};

export default DocumentUpload;