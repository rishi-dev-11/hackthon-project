import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Box,
  Paper,
  Button
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import {
  Upload as UploadIcon,
  History as HistoryIcon,
  Analytics as AnalyticsIcon,
  FormatPaint as FormatPaintIcon,
  Stars as StarsIcon
} from '@mui/icons-material';
import AnimatedCard from '../components/AnimatedCard';
import AnimatedProgress from '../components/AnimatedProgress';
import AnimatedNotification from '../components/AnimatedNotification';
import UsageStatistics from '../components/UsageStatistics';
import FeatureGuard from '../components/FeatureGuard';
import UsageLimit from '../components/UsageLimit';
import { useSubscription } from '../hooks/useSubscription';
import { useUser } from '../hooks/useUser';

const features = [
  {
    title: "Smart Document Upload",
    description: "Upload any document format and let AI handle the rest",
    icon: UploadIcon,
    path: "/upload"
  },
  {
    title: "Template Management",
    description: "Create and manage your document templates",
    icon: FormatPaintIcon,
    path: "/templates"
  },
  {
    title: "Document History",
    description: "View and manage your past transformations",
    icon: HistoryIcon,
    path: "/history"
  },
  {
    title: "Analytics Dashboard",
    description: "Track your document processing metrics",
    icon: AnalyticsIcon,
    path: "/analytics"
  }
];

function Dashboard() {
  const [showNotification, setShowNotification] = useState(false);
  const navigate = useNavigate();
  const { getUserSubscription } = useUser();
  const { usageStats } = useSubscription();
  const currentSubscription = getUserSubscription();

  const handleCardClick = (path) => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
    navigate(path);
  };

  return (
    <Box sx={{ 
      width: '100vw',
      minHeight: '100vh',
      py: { xs: 2, sm: 4, md: 6 },
      px: { xs: 1, sm: 2, md: 3 },
      background: 'linear-gradient(145deg, #f5f7fa 0%, #e4e8eb 100%)',
      overflow: 'hidden',
      position: 'relative',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Container 
        maxWidth={false} 
        sx={{ 
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: { xs: 3, sm: 4, md: 6 }
        }}
      >
        <Motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ 
            textAlign: 'center', 
            mb: { xs: 3, sm: 4, md: 6 },
            position: 'relative'
          }}>
            <Typography 
              variant="h3" 
              gutterBottom 
              sx={{ 
                fontWeight: 'bold',
                fontSize: { xs: '2rem', sm: '2.5rem', md: '3rem' },
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                backgroundClip: 'text',
                textFillColor: 'transparent',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              Welcome to DocuMorph AI
            </Typography>
            <Typography 
              variant="h6" 
              color="text.secondary"
              sx={{ 
                mb: { xs: 2, sm: 3, md: 4 },
                fontSize: { xs: '1rem', sm: '1.25rem', md: '1.5rem' }
              }}
            >
              Transform your documents with AI-powered formatting
            </Typography>
          </Box>
        </Motion.div>

        <Grid 
          container 
          spacing={{ xs: 2, sm: 3, md: 4 }}
          sx={{ 
            width: '100%',
            m: 0,
            flexGrow: 1
          }}
        >
          {features.map((feature, index) => (
            <Grid 
              item 
              xs={12} 
              sm={6} 
              md={6} 
              lg={3} 
              key={feature.title}
              sx={{ 
                display: 'flex',
                height: { xs: 'auto', sm: '100%' }
              }}
            >
              <AnimatedCard
                title={feature.title}
                description={feature.description}
                icon={feature.icon}
                onClick={() => handleCardClick(feature.path)}
                delay={index * 0.1}
                sx={{ 
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              />
            </Grid>
          ))}
        </Grid>

        <Paper 
          elevation={0}
          sx={{ 
            mt: { xs: 3, sm: 4, md: 6 },
            p: { xs: 2, sm: 3, md: 4 },
            borderRadius: 4,
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            width: '100%'
          }}
        >
          <Typography 
            variant="h5" 
            gutterBottom 
            sx={{ 
              fontWeight: 600,
              fontSize: { xs: '1.25rem', sm: '1.5rem', md: '1.75rem' }
            }}
          >
            Processing Status
          </Typography>
          <Box sx={{ 
            display: 'flex',
            flexDirection: 'column',
            gap: { xs: 2, sm: 3 }
          }}>
            <AnimatedProgress progress={75} label="Document Analysis" />
            <AnimatedProgress progress={45} label="Formatting" />
            <AnimatedProgress progress={90} label="Quality Check" />
          </Box>
        </Paper>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            {/* Main dashboard content */}
          </Grid>
          
          <Grid item xs={12} md={4}>
            <UsageStatistics />
            
            {currentSubscription === 'free' && (
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 3, 
                  borderRadius: 2, 
                  mt: 3,
                  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  color: 'white'
                }}
              >
                <Typography variant="h6" gutterBottom>
                  Upgrade to Premium
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  Get unlimited access to all features and premium templates
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<StarsIcon />}
                  onClick={() => navigate('/upgrade')}
                  sx={{
                    bgcolor: 'white',
                    color: 'primary.main',
                    '&:hover': {
                      bgcolor: 'rgba(255,255,255,0.9)',
                    }
                  }}
                >
                  View Premium Benefits
                </Button>
              </Paper>
            )}

            <Box sx={{ mt: 3 }}>
              <FeatureGuard feature="templates">
                {/* Premium features section */}
              </FeatureGuard>
            </Box>
          </Grid>
        </Grid>

        {showNotification && (
          <AnimatedNotification 
            message="Processing your document..." 
            severity="info"
          />
        )}
      </Container>
    </Box>
  );
}

export default Dashboard;