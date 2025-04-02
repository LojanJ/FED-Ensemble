/* eslint-disable react/prop-types */
import { AppBar, Toolbar, Box, Typography, Chip, IconButton } from '@mui/material';
import { BarChart as BarChartIcon } from '@mui/icons-material';
import DarkModeToggle from '../../Utilities';

const AppHeader = ({ darkMode, setDarkMode, lastUpdated, isLoading, connectionStatus }) => {
  return (
    <AppBar 
      position="fixed" 
      elevation={2}
      color="primary"
    >
      <Toolbar sx={{ minHeight: { xs: '64px', md: '70px' }, px: { xs: 2, md: 3 } }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box
            sx={{ 
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 40,
              height: 40,
              bgcolor: 'primary.dark',
              borderRadius: 1.5,
              mr: 2,
              color: 'white'
            }}
          >
            <BarChartIcon fontSize="medium" />
          </Box>
          
          <Box>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                fontSize: { xs: '1.1rem', sm: '1.2rem', md: '1.3rem' },
                lineHeight: 1.2,
              }}
            >
              FED-Ensemble
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                opacity: 0.9, 
                display: { xs: 'none', sm: 'block' }
              }}
            >
              Federated Learning System
            </Typography>
          </Box>
        </Box>

        <Box sx={{ flexGrow: 1 }} />
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip 
            label={
              isLoading ? "Syncing..." : 
              connectionStatus === 'active' ? "System Active" : 
              connectionStatus === 'connecting' ? "Connecting..." :
              "Connection Error"
            } 
            size="small" 
            color={
              isLoading ? "warning" : 
              connectionStatus === 'active' ? "success" : 
              connectionStatus === 'connecting' ? "info" :
              "error"
            }
            sx={{ 
              height: 26, 
              '& .MuiChip-label': { px: 1.5, fontWeight: 500 },
              display: { xs: 'none', sm: 'flex' }
            }}
          />
          
          <Typography 
            variant="body2" 
            sx={{ 
              display: { xs: 'none', md: 'block' },
              mr: 1
            }}
          >
            Last updated: {lastUpdated.toLocaleTimeString()}
          </Typography>
          
          <IconButton 
            color='inherit' 
            onClick={() => setDarkMode(!darkMode)} 
            sx={{ borderRadius: 1 }}
          > 
            <DarkModeToggle />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default AppHeader; 