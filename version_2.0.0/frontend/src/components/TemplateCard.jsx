import React from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Button,
  Chip,
} from '@mui/material';
import { Stars as StarsIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { motion as Motion } from 'framer-motion';
import { useUser } from '../hooks/useUser';

const TemplateCard = ({ template, isPremium }) => {
  const navigate = useNavigate();
  const { getUserSubscription } = useUser();
  const currentSubscription = getUserSubscription();
  const canUseTemplate = !isPremium || currentSubscription === 'premium';

  return (
    <Motion.div
      whileHover={{ y: -5 }}
      transition={{ duration: 0.2 }}
    >
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          borderRadius: 2,
          overflow: 'visible',
          boxShadow: isPremium ? '0 4px 20px rgba(33, 150, 243, 0.15)' : '0 2px 10px rgba(0,0,0,0.08)',
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
              '& .MuiChip-icon': {
                color: 'inherit',
              },
            }}
          />
        )}

        <CardMedia
          component="img"
          height="160"
          image={template.thumbnailUrl || '/placeholder-template.png'}
          alt={template.title}
          sx={{
            objectFit: 'cover',
            filter: !canUseTemplate ? 'grayscale(50%)' : 'none',
          }}
        />

        <CardContent sx={{ flexGrow: 1, pb: 2 }}>
          <Typography variant="h6" gutterBottom component="div">
            {template.title}
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            {template.description}
          </Typography>
        </CardContent>

        <Box sx={{ p: 2, pt: 0 }}>
          {canUseTemplate ? (
            <Button
              fullWidth
              variant="contained"
              onClick={() => navigate(`/templates/${template.id}`)}
              sx={{
                background: isPremium 
                  ? 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)'
                  : undefined,
                boxShadow: isPremium 
                  ? '0 3px 5px 2px rgba(33, 203, 243, .3)'
                  : undefined,
              }}
            >
              Use Template
            </Button>
          ) : (
            <Button
              fullWidth
              variant="outlined"
              color="primary"
              onClick={() => navigate('/upgrade')}
              startIcon={<StarsIcon />}
            >
              Upgrade to Use
            </Button>
          )}
        </Box>
      </Card>
    </Motion.div>
  );
};

export default TemplateCard;