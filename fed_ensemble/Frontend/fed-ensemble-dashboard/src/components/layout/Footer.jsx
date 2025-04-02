/* eslint-disable react/prop-types */
import { Box, Container, Typography } from '@mui/material';

const Footer = ({ connectionStatus }) => {
  return (
    <Box 
      component="footer" 
      sx={{ 
        py: 1.5, 
        mt: 'auto', 
        bgcolor: 'background.paper', 
        borderTop: 1, 
        borderColor: 'divider'
      }}
    >
      <Container>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
          <Typography variant="caption" color="text.secondary">
            FED-Ensemble v1.0.3
          </Typography>
          <Box 
            sx={{ 
              width: 6, 
              height: 6, 
              borderRadius: '50%', 
              bgcolor: connectionStatus === 'active' ? 'success.main' : 'error.main' 
            }} 
          />
        </Box>
      </Container>
    </Box>
  );
};

export default Footer; 