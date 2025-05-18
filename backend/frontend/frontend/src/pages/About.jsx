// import React from 'react';
// import {
//   Container,
//   Typography,
//   Grid,
//   Box,
//   Card,
//   CardContent,
//   useTheme,
// } from '@mui/material';
// import { motion } from 'framer-motion';
// import {
//   Security,
//   Speed,
//   AutoAwesome,
//   Support,
// } from '@mui/icons-material';

// const features = [
//   {
//     icon: <Security sx={{ fontSize: 40 }} />,
//     title: 'Secure Processing',
//     description: 'Your documents are processed with enterprise-grade security and encryption.',
//   },
//   {
//     icon: <Speed sx={{ fontSize: 40 }} />,
//     title: 'Fast Processing',
//     description: 'Quick and efficient document processing with real-time updates.',
//   },
//   {
//     icon: <AutoAwesome sx={{ fontSize: 40 }} />,
//     title: 'Smart Templates',
//     description: 'Intelligent template system that adapts to your needs.',
//   },
//   {
//     icon: <Support sx={{ fontSize: 40 }} />,
//     title: '24/7 Support',
//     description: 'Round-the-clock customer support for all your needs.',
//   },
// ];

// const About = () => {
//   const theme = useTheme();

//   return (
//     <Container maxWidth="lg">
//       <Box sx={{ py: 8 }}>
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.5 }}
//         >
//           <Typography
//             variant="h2"
//             align="center"
//             gutterBottom
//             sx={{
//               fontWeight: 'bold',
//               background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.light} 90%)`,
//               WebkitBackgroundClip: 'text',
//               WebkitTextFillColor: 'transparent',
//             }}
//           >
//             About Us
//           </Typography>
//           <Typography
//             variant="h5"
//             align="center"
//             color="text.secondary"
//             sx={{ mb: 6 }}
//           >
//             Transforming document processing with AI-powered solutions
//           </Typography>
//         </motion.div>

//         <Grid container spacing={4}>
//           {features.map((feature, index) => (
//             <Grid item xs={12} md={6} key={index}>
//               <motion.div
//                 initial={{ opacity: 0, y: 20 }}
//                 animate={{ opacity: 1, y: 0 }}
//                 transition={{ duration: 0.5, delay: index * 0.1 }}
//               >
//                 <Card
//                   sx={{
//                     height: '100%',
//                     borderRadius: theme.shape.borderRadius * 2,
//                     transition: 'transform 0.3s, box-shadow 0.3s',
//                     '&:hover': {
//                       transform: 'translateY(-8px)',
//                       boxShadow: theme.shadows[8],
//                     },
//                   }}
//                 >
//                   <CardContent sx={{ p: 4 }}>
//                     <Box
//                       sx={{
//                         color: theme.palette.primary.main,
//                         mb: 2,
//                       }}
//                     >
//                       {feature.icon}
//                     </Box>
//                     <Typography variant="h5" gutterBottom fontWeight="bold">
//                       {feature.title}
//                     </Typography>
//                     <Typography color="text.secondary">
//                       {feature.description}
//                     </Typography>
//                   </CardContent>
//                 </Card>
//               </motion.div>
//             </Grid>
//           ))}
//         </Grid>
//       </Box>
//     </Container>
//   );
// };

// export default About;

import React from 'react';
import {
  Container,
  Typography,
  Grid,
  Box,
  Card,
  CardContent,
  useTheme,
} from '@mui/material';
import { motion as Motion } from 'framer-motion';
import {
  Security,
  Speed,
  AutoAwesome,
  Support,
} from '@mui/icons-material';

const features = [
  {
    icon: <Security sx={{ fontSize: 40 }} />,
    title: 'Secure Processing',
    description: 'Your documents are processed with enterprise-grade security and encryption.',
  },
  {
    icon: <Speed sx={{ fontSize: 40 }} />,
    title: 'Fast Processing',
    description: 'Quick and efficient document processing with real-time updates.',
  },
  {
    icon: <AutoAwesome sx={{ fontSize: 40 }} />,
    title: 'Smart Templates',
    description: 'Intelligent template system that adapts to your needs.',
  },
  {
    icon: <Support sx={{ fontSize: 40 }} />,
    title: '24/7 Support',
    description: 'Round-the-clock customer support for all your needs.',
  },
];

const About = () => {
  const theme = useTheme();

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 8 }}>
        <Motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography
            variant="h2"
            align="center"
            gutterBottom
            sx={{
              fontWeight: 'bold',
              background: `linear-gradient(45deg, ${theme.palette.primary.main} 30%, ${theme.palette.primary.light} 90%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            About Us
          </Typography>
          <Typography
            variant="h5"
            align="center"
            color="text.secondary"
            sx={{ mb: 6 }}
          >
            Transforming document processing with AI-powered solutions
          </Typography>
        </Motion.div>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    borderRadius: theme.shape.borderRadius * 2,
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: theme.shadows[8],
                    },
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Box
                      sx={{
                        color: theme.palette.primary.main,
                        mb: 2,
                      }}
                    >
                      {feature.icon}
                    </Box>
                    <Typography variant="h5" gutterBottom fontWeight="bold">
                      {feature.title}
                    </Typography>
                    <Typography color="text.secondary">
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Motion.div>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Container>
  );
};

export default About;