import React, { useState, useEffect, useRef } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Box,
  Paper,
  Button,
  Divider,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  CircularProgress,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import AddIcon from '@mui/icons-material/Add';
import DescriptionIcon from '@mui/icons-material/Description';
import StyleIcon from '@mui/icons-material/Style';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import TableChartIcon from '@mui/icons-material/TableChart';
import UpgradeIcon from '@mui/icons-material/Upgrade';
import gsap from 'gsap';
import AnimatedCard from '../components/AnimatedCard';
import AnimatedProgress from '../components/AnimatedProgress';
import AnimatedNotification from '../components/AnimatedNotification';
import UsageStatistics from '../components/UsageStatistics';
import FeatureGuard from '../components/FeatureGuard';
import UsageLimit from '../components/UsageLimit';
import { useSubscription } from '../hooks/useSubscription';
import { useUser } from '../hooks/useUser';

// Helper function to get icon by file type
const getFileIcon = (fileType) => {
  const iconMap = {
    'pdf': <DescriptionIcon color="primary" fontSize="large" />,
    'docx': <DescriptionIcon color="secondary" fontSize="large" />,
    'txt': <DescriptionIcon color="action" fontSize="large" />,
    'default': <DescriptionIcon color="disabled" fontSize="large" />
  };
  
  return iconMap[fileType] || iconMap.default;
};

// Sample data - replace with actual API calls
const recentDocuments = [
  { id: 1, name: 'Research Paper.docx', type: 'docx', date: '2023-06-12', status: 'completed', size: '1.2 MB' },
  { id: 2, name: 'Conference Presentation.pdf', type: 'pdf', date: '2023-06-10', status: 'completed', size: '2.8 MB' },
  { id: 3, name: 'Project Notes.txt', type: 'txt', date: '2023-06-08', status: 'processing', size: '512 KB' },
  { id: 4, name: 'Draft Manuscript.docx', type: 'docx', date: '2023-06-05', status: 'error', size: '1.5 MB' }
];

// Sample usage stats
const usageStats = {
  documentsProcessed: 18,
  documentsLimit: 25,
  templatesUsed: 7,
  templatesLimit: 12,
  aiEnhancements: 12,
  aiEnhancementsLimit: 20,
  storageUsed: 25.8,
  storageLimit: 100
};

// Sample activity data
const activityData = [
  { 
    type: 'upload', 
    message: 'Research Paper.docx was uploaded', 
    timestamp: '2 hours ago',
    icon: <AnalyticsIcon color="primary" />
  },
  { 
    type: 'process', 
    message: 'Conference Presentation.pdf was processed', 
    timestamp: '6 hours ago',
    icon: <AnalyticsIcon color="success" />
  },
  { 
    type: 'template', 
    message: 'Academic template was applied to Project Notes', 
    timestamp: '1 day ago',
    icon: <StyleIcon color="secondary" />
  },
  { 
    type: 'download', 
    message: 'Draft Manuscript.docx was downloaded', 
    timestamp: '2 days ago',
    icon: <AnalyticsIcon color="primary" />
  }
];

// Recent document component
const RecentDocument = ({ document, onClick }) => {
  const docRef = useRef(null);
  
  useEffect(() => {
    gsap.fromTo(
      docRef.current,
      { opacity: 0, y: 20 },
      { 
        opacity: 1, 
        y: 0, 
        duration: 0.4, 
        delay: 0.1,
        ease: "power2.out"
      }
    );
  }, []);
  
  return (
    <div ref={docRef}>
      <Paper
        sx={{
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: 3
          }
        }}
        onClick={() => onClick(document.id)}
      >
        {getFileIcon(document.type)}
        <Typography variant="subtitle1" sx={{ mt: 1, textAlign: 'center' }}>
          {document.name}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {document.size} • {document.date}
        </Typography>
      </Paper>
    </div>
  );
};

// Feature card component
const FeatureCard = ({ feature, onClick }) => {
  const cardRef = useRef(null);
  
  useEffect(() => {
    gsap.fromTo(
      cardRef.current,
      { opacity: 0, y: 20 },
      { 
        opacity: 1, 
        y: 0, 
        duration: 0.5, 
        delay: feature.delay || 0,
        ease: "power2.out"
      }
    );
  }, [feature.delay]);
  
  return (
    <div ref={cardRef}>
      <Paper
        sx={{
          p: 3,
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          transition: 'all 0.3s',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            transform: 'translateY(-8px)',
            boxShadow: 4
          },
          ...(feature.premium && !feature.available && {
            opacity: 0.7,
            '&:after': {
              content: '"Premium"',
              position: 'absolute',
              top: 10,
              right: 10,
              fontSize: '0.7rem',
              backgroundColor: 'warning.main',
              color: 'warning.contrastText',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 'bold'
            }
          })
        }}
        onClick={onClick}
      >
        <IconButton
          sx={{ 
            mb: 2,
            color: feature.iconColor || 'primary.main',
            backgroundColor: `${feature.iconColor || 'primary.main'}10`,
            p: 1.5
          }}
        >
          {feature.icon}
        </IconButton>
        <Typography variant="h6" align="center" gutterBottom>
          {feature.title}
        </Typography>
        <Typography variant="body2" align="center" color="text.secondary">
          {feature.description}
        </Typography>
      </Paper>
    </div>
  );
};

function Dashboard() {
  const [showNotification, setShowNotification] = useState(false);
  const navigate = useNavigate();
  const { getUserSubscription } = useUser();
  const { usageStats: subscriptionStats } = useSubscription();
  const currentSubscription = getUserSubscription();
  const isPremium = currentSubscription === 'premium';
  const containerRef = useRef(null);
  
  useEffect(() => {
    // Check if it's the first visit
    const hasVisitedDashboard = localStorage.getItem('hasVisitedDashboard');
    if (!hasVisitedDashboard) {
      setShowNotification(true);
      localStorage.setItem('hasVisitedDashboard', 'true');
    }
    
    // GSAP animation for the container
    gsap.fromTo(
      containerRef.current,
      { opacity: 0 },
      { opacity: 1, duration: 0.8, ease: "power1.out" }
    );
  }, []);

  const handleUploadClick = () => {
    navigate('/upload');
  };

  const handleTemplatesClick = () => {
    navigate('/templates');
  };

  const handleDocumentClick = (documentId) => {
    navigate(`/preview/${documentId}`);
  };

  const handleCardClick = (path) => {
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
    navigate(path);
  };

  const featureCards = [
    {
      title: "Upload Document",
      description: "Upload any document format for AI processing",
      icon: <AddIcon fontSize="large" />,
      path: "/upload",
      color: "primary.main",
      delay: 0.1,
      available: true,
      premium: false
    },
    {
      title: "Templates",
      description: "Browse and apply professional document templates",
      icon: <StyleIcon fontSize="large" />,
      path: "/templates",
      color: "#9C27B0",
      delay: 0.2,
      available: true,
      premium: false
    },
    {
      title: "Style Analysis",
      description: "Analyze and enhance your document's style",
      icon: <StyleIcon fontSize="large" />,
      path: "/style",
      color: "#9c27b0",
      delay: 0.3,
      available: isPremium,
      premium: true
    },
    {
      title: "Content Analysis",
      description: "Deep analysis of your document's content",
      icon: <AnalyticsIcon fontSize="large" />,
      path: "/analysis",
      color: "#ff9800",
      delay: 0.4,
      available: isPremium,
      premium: true
    },
    {
      title: "Table Management",
      description: "Extract and manage tables in your documents",
      icon: <TableChartIcon fontSize="large" />,
      path: "/tables",
      color: "#f44336",
      delay: 0.5,
      available: isPremium,
      premium: true
    },
    {
      title: "Upgrade",
      description: "Upgrade to Premium for additional features",
      icon: <UpgradeIcon fontSize="large" />,
      path: "/upgrade",
      color: "#ffc107",
      delay: 0.6,
      available: true,
      premium: false
    }
  ];

  return (
    <div ref={containerRef}>
      <Container 
        maxWidth="lg" 
        sx={{ 
          py: { xs: 2, sm: 4, md: 5 },
          px: { xs: 1, sm: 2, md: 3 },
        }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography 
            variant="h4" 
            gutterBottom 
            sx={{ 
              fontWeight: 600,
              color: 'primary.dark'
            }}
          >
            Dashboard
          </Typography>
          <Typography 
            variant="body1" 
            color="text.secondary"
            sx={{ mb: 2 }}
          >
            Welcome back! Manage your documents and access DocuMorph AI features.
          </Typography>
        </Box>

        <UsageStatistics />
        
        <Box sx={{ mt: 6, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom fontWeight="bold">
            Quick Actions
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          {featureCards.map((card, index) => (
            <Grid 
              xs={12} 
              sm={6} 
              md={4} 
              key={card.title}
            >
              <FeatureCard 
                feature={card} 
                onClick={() => card.available ? handleCardClick(card.path) : handleCardClick('/upgrade')} 
              />
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 6, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom fontWeight="bold">
            Recent Documents
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          {recentDocuments.map((doc) => (
            <Grid item xs={12} sm={6} md={3} key={doc.id}>
              <RecentDocument document={doc} onClick={handleDocumentClick} />
            </Grid>
          ))}
        </Grid>
      </Container>

      {showNotification && (
        <AnimatedNotification 
          message="Welcome to your dashboard! Upload a document or browse templates to get started." 
          severity="info"
          onClose={() => setShowNotification(false)}
        />
      )}
    </div>
  );
}

export default Dashboard;