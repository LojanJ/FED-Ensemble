/* eslint-disable react/prop-types */
import { Card, CardContent, Table, TableBody, TableCell, TableContainer, TableRow, Typography, Box, Button } from '@mui/material';
import { Settings as SettingsIcon, Refresh as RefreshIcon } from '@mui/icons-material';
import { useState } from 'react';
import { fetchConfig } from '../../utils/api';

const ConfigPanel = ({ config }) => {
  const [loading, setLoading] = useState(false);

  const handleRefreshConfig = async () => {
    setLoading(true);
    try {
      // This is just to simulate a refresh attempt - in a real app, 
      // you would handle this via a central state or context
      await fetchConfig();
      window.location.reload(); // Simple refresh for demonstration
    } catch (error) {
      console.error("Failed to refresh config:", error);
    } finally {
      setLoading(false);
    }
  };

  // Check if config is empty, null, or has no properties
  const isConfigEmpty = !config || Object.keys(config).length === 0;

  if (isConfigEmpty) {
    return (
      <Card>
        <CardContent>
          <Typography variant="subtitle1" display="flex" alignItems="center" gap={1}>
            <SettingsIcon fontSize="small" /> System Configuration
          </Typography>
          <Box sx={{ 
            my: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center',
            gap: 2
          }}>
            <Typography variant="body2" align="center" color="text.secondary">
              No configuration data available
            </Typography>
            <Button 
              startIcon={<RefreshIcon />} 
              variant="outlined" 
              size="small" 
              onClick={handleRefreshConfig}
              disabled={loading}
            >
              {loading ? 'Refreshing...' : 'Refresh Configuration'}
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" display="flex" alignItems="center" gap={1}>
            <SettingsIcon fontSize="small" /> System Configuration
          </Typography>
          <Button 
            startIcon={<RefreshIcon />} 
            variant="text" 
            size="small" 
            onClick={handleRefreshConfig}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        <TableContainer>
          <Table size="small">
            <TableBody>
              {Object.entries(config).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell component="th" scope="row">
                    <Typography variant="body2" fontWeight="medium">
                      {key}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {Array.isArray(value) 
                        ? (value.length > 0 ? value.join(', ') : '[]')
                        : value === null 
                          ? 'null' 
                          : typeof value === 'object' 
                            ? JSON.stringify(value) 
                            : String(value)
                      }
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default ConfigPanel; 