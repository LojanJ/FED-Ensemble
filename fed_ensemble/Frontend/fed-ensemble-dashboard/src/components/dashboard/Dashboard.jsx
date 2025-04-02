/* eslint-disable react/prop-types */
import { useState } from 'react';
import { Box, Container, Card, Grid, Tabs, Tab, Paper, Typography } from '@mui/material';
import { BarChart as BarChartIcon, Security as SecurityIcon } from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

// Import custom components
import { useDataFetching } from '../../hooks/useDataFetching';
import { useThemeContext } from '../../context/ThemeContext';
import { computeMean } from '../../utils/dataHelpers';
import AppHeader from '../layout/AppHeader';
import Footer from '../layout/Footer';
import MetricsGrid from '../metrics/MetricsGrid';
import AccuracyChart from '../charts/AccuracyChart';
import LossChart from '../charts/LossChart';
import ClientsTable from '../tables/ClientsTable';
import SystemHealthPanel from './SystemHealthPanel';
import ConfigPanel from './ConfigPanel';

const Dashboard = () => {
  // Get theme context
  const { darkMode, setDarkMode } = useThemeContext();
  
  // Use data fetching hook
  const { 
    metrics, 
    config, 
    processedData, 
    lastUpdated, 
    isLoading, 
    connectionStatus 
  } = useDataFetching();

  // Local states
  const [activeTab, setActiveTab] = useState(0);

  // Calculate anomaly threshold - using mean of all thresholds
  const anomalyThreshold = computeMean(metrics?.training_progress?.map(progress => progress.threshold)) || 0;

  // Calculate anomaly level counts using clients data
  const calculateAnomalyCounts = () => {
    const lowAnomalies = metrics?.clients?.filter(client => {
      const scores = client?.anomaly_scores || [];
      const mean = scores.reduce((sum, val) => sum + parseFloat(val || 0), 0) / scores.length;
      return mean <= 0.85 * anomalyThreshold;
    })?.length || 0;
    
    const mediumAnomalies = metrics?.clients?.filter(client => {
      const scores = client?.anomaly_scores || [];
      const mean = scores.reduce((sum, val) => sum + parseFloat(val || 0), 0) / scores.length;
      return mean > 0.85 * anomalyThreshold && mean <= anomalyThreshold;
    })?.length || 0;
    
    const highAnomalies = metrics?.clients?.filter(client => {
      const scores = client?.anomaly_scores || [];
      const mean = scores.reduce((sum, val) => sum + parseFloat(val || 0), 0) / scores.length;
      return mean > anomalyThreshold;
    })?.length || 0;

    return [
      { name: 'Low', value: lowAnomalies },
      { name: 'Medium', value: mediumAnomalies },
      { name: 'High', value: highAnomalies }
    ];
  };

  const anomalyData = calculateAnomalyCounts();
  const ANOMALY_COLORS = ['#4CAF50', '#FF9800', '#F44336'];

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ 
      bgcolor: 'background.default', 
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <AppHeader 
        darkMode={darkMode} 
        setDarkMode={setDarkMode} 
        lastUpdated={lastUpdated} 
        isLoading={isLoading} 
        connectionStatus={connectionStatus} 
      />
      
      {/* Main content */}
      <Container maxWidth="lg" style={{paddingTop:'64px'}} sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        {/* Top metrics cards */}
        <MetricsGrid metrics={metrics} config={config} />

        {/* Training Progress Chart */}
        <Box sx={{ display: 'flex', gap: 2, marginBottom: 4 }}>
          <AccuracyChart processedData={processedData} />
          <LossChart processedData={processedData} />
        </Box>

        {/* Tabs for Client Performance and Security */}
        <Card sx={{ mb: 4 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange} 
              aria-label="dashboard tabs"
              indicatorColor="primary"
              textColor="primary"
            >
              <Tab label="Distributed Performance Chart" icon={<BarChartIcon />} iconPosition="start" />
              <Tab label="Security Monitoring" icon={<SecurityIcon />} iconPosition="start" />
            </Tabs>
          </Box>
          
          <Box sx={{ p: 3 }}>
            {activeTab === 0 && (
              <Box>
                <ClientsTable clients={metrics.clients} anomalyThreshold={anomalyThreshold} />
              </Box>
            )}
            
            {activeTab === 1 && (
              <Box>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Malicious Client Distribution
                      </Typography>
                      <Box sx={{ height: 300 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={anomalyData}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              outerRadius={80}
                              dataKey="value"
                            >
                              {anomalyData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={ANOMALY_COLORS[index % ANOMALY_COLORS.length]} />
                              ))}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                      </Box>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Anomaly Threshold
                      </Typography>
                      <Box sx={{ height: 300, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                        <Typography variant="h3" color="primary">
                          {anomalyThreshold.toFixed(3)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                          Current Anomaly Detection Threshold (Mean)
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Box>
        </Card>
        
        {/* System Health and Config */}
        <Grid container spacing={3}>
          {/* System Health */}
          <Grid item xs={12} md={6}>
            <SystemHealthPanel performance={metrics.performance} />
          </Grid>

          {/* Model Configuration */}
          <Grid item xs={12} md={6}>
            <ConfigPanel config={config} />
          </Grid>
        </Grid>
      </Container>

      {/* Footer */}
      <Footer connectionStatus={connectionStatus} />
    </Box>
  );
};

export default Dashboard; 