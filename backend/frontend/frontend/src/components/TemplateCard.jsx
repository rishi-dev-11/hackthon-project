import React, { useState, useContext } from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Button,
  Chip,
  Skeleton,
} from '@mui/material';
import { Stars as StarsIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import { UserContext } from '../context/UserContext'; // ✅ Correct context usage

const TemplateCard = ({ template }) => {
  const navigate = useNavigate();
  const userContext = useContext(UserContext);
  const [isLoading, setIsLoading] = useState(false);

  const subscriptionType = userContext?.subscription || 'free'; // ✅ Replaces getUserSubscription
  const isPremium = template?.isPremium;
  const hasAccess = !isPremium || subscriptionType === 'premium';

  const handleTemplateClick = async () => {
    if (!hasAccess) {
      navigate('/upgrade');
      return;
    }

    try {
      setIsLoading(true);
      navigate(`/template/${template.id}`);
    } catch (error) {
      console.error('Error selecting template:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!template) {
    return (
      <Card sx={{ height: '100%', minHeight: 300 }}>
        <Skeleton variant="rectangular" height={140} />
        <CardContent>
          <Skeleton variant="text" />
          <Skeleton variant="text" width="60%" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Motion.div whileHover={{ y: -5 }} transition={{ duration: 0.2 }}>
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          borderRadius: 2,
          overflow: 'visible',
          boxShadow: isPremium
            ? '0 4px 20px rgba(33, 150, 243, 0.15)'
            : '0 2px 10px rgba(0,0,0,0.08)',
          border: '1px solid',
          borderColor: isPremium ? 'primary.main' : 'divider',
        }}
      >
        {isPremium && (
          <Chip
            icon={<StarsIcon />}
            label="Premium"
            color="primary"
            size="small"
            sx={{
              position: 'absolute',
              top: -12,
              right: 12,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              color: 'white',
              zIndex: 1,
            }}
          />
        )}

        <CardMedia
          component="img"
          height="140"
          image={template.imageUrl || '/placeholder-template.png'}
          alt={template.name || 'Template Preview'}
          sx={{
            objectFit: 'cover',
            opacity: hasAccess ? 1 : 0.5,
          }}
        />

        <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Typography gutterBottom variant="h6" component="h2">
            {template.name}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {template.description}
          </Typography>

          <Box sx={{ mt: 'auto' }}>
            <Button
              fullWidth
              variant={hasAccess ? 'contained' : 'outlined'}
              onClick={handleTemplateClick}
              disabled={isLoading}
              color={hasAccess ? 'primary' : 'inherit'}
            >
              {isLoading
                ? 'Loading...'
                : hasAccess
                ? 'Use Template'
                : 'Upgrade to Use'}
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Motion.div>
  );
};

export default TemplateCard;
