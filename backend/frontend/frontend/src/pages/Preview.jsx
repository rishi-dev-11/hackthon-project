import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Container, Typography, Box, Button, CircularProgress, Alert } from '@mui/material';
import { getPreview, downloadDocument } from '../services/api';

function Preview() {
  const { documentId } = useParams();
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchPreview = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getPreview(documentId);
        setPreview(data.formatted_content || data.content || '');
      } catch (err) {
        console.error('Preview error:', err);
        setError('Failed to load preview.');
      } finally {
        setLoading(false);
      }
    };
    fetchPreview();
  }, [documentId]);

  const handleDownload = async () => {
    setDownloading(true);
    setError(null);
    try {
      const blob = await downloadDocument(documentId);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'formatted_document.docx');
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
    } catch (err) {
      console.error('Download error:', err);
      setError('Download failed.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Document Preview
      </Typography>
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">{error}</Alert>
      ) : (
        <Box sx={{ border: '1px solid #eee', borderRadius: 2, p: 3, bgcolor: '#fafafa', mb: 2 }}>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{preview}</pre>
        </Box>
      )}
      <Button
        variant="contained"
        color="primary"
        onClick={handleDownload}
        disabled={downloading || loading}
      >
        {downloading ? <CircularProgress size={24} /> : 'Download'}
      </Button>
    </Container>
  );
}

export default Preview; 