/* eslint-disable react/prop-types */
import { Box, Grid } from '@mui/material';
import { 
  Group as GroupIcon,
  BugReport as BugReportIcon,
  Dataset as DatasetIcon,
  Speed as SpeedIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import MetricCard from './MetricCard';

const MetricsGrid = ({ metrics, config }) => {
  // Function to calculate detection accuracy - using the original approach
  const calculateDetectionAccuracy = () => {
    if (!metrics?.training_progress || metrics.training_progress.length <= 1) return 'N/A';
    
    return parseFloat(metrics.training_progress.reduce((acc, curr) => {
      const dacc = acc + curr.detection_accuracy / (metrics.training_progress.length - 1);
      if (isNaN(dacc)) return acc;
      else return dacc;
    }, 0)).toFixed(2);
  };

  // Calculate poisoning ratio safely
  const calculatePoisoningRatio = () => {
    if (!config?.malicious_clients_id || !metrics?.clients?.length) {
      return 'N/A';
    }
    
    const maliciousCount = Array.isArray(config.malicious_clients_id) 
      ? config.malicious_clients_id.length 
      : 0;
    
    if (maliciousCount === 0 || metrics.clients.length === 0) {
      return '0%';
    }
    
    return (maliciousCount / metrics.clients.length).toFixed(2);
  };
  
  // Define the metric cards data
  const metricCardsData = [
    {
      title: "Number of Client Participants",
      value: metrics.clients?.length || 'N/A',
      icon: <GroupIcon />,
      color: 'primary',
      description: "Active clients in federation",
      fontWeight: '600',
      showProgress: false
    },
    {
      title: "Poisoning Method",
      value: config?.attack_type ? String(config.attack_type).replace('_', ' ') : 'N/A',
      icon: <BugReportIcon />,
      color: 'warning',
      description: "Current attack simulation type",
      fontWeight: '600',
      showProgress: false
    },
    {
      title: "Dataset",
      value: config?.dataset ? String(config.dataset).toUpperCase() : 'N/A',
      icon: <DatasetIcon />,
      color: 'info',
      description: "Training dataset in use",
      fontWeight: '600',
      showProgress: false
    },
    {
      title: "Simulated Poisoning Ratio",
      value: calculatePoisoningRatio(),
      icon: <SpeedIcon />,
      color: 'error',
      description: "Percent of malicious clients",
      fontWeight: '700',
      showProgress: false
    },
    {
      title: "Detection Accuracy",
      value: calculateDetectionAccuracy(),
      icon: <AnalyticsIcon />,
      color: 'secondary',
      description: "Accuracy of anomaly detection system",
      fontWeight: '700',
      showProgress: true
    }
  ];

  return (
    <Box sx={{ position: 'relative', mb: 4, mt: 1 }}>
      <Grid container spacing={2.5}>
        {/* First row - 3 cards with different sizes */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard {...metricCardsData[0]} />
        </Grid>

        {/* Dataset & Poisoning Method - Mid-sized cards */}
        <Grid item xs={12} sm={6} md={4} sx={{ mt: { xs: 0, md: 0 } }}>
          <MetricCard {...metricCardsData[1]} />
        </Grid>

        <Grid item xs={12} sm={6} md={5} sx={{ mt: { xs: 0, md: 0 } }}>
          <MetricCard {...metricCardsData[2]} />
        </Grid>

        {/* Second row - 2 prominent cards with key metrics */}
        <Grid item xs={12} sm={6} md={6} sx={{ mt: { xs: 0, md: -0.5 } }}>
          <MetricCard {...metricCardsData[3]} />
        </Grid>

        <Grid item xs={12} sm={6} md={6} sx={{ mt: { xs: 0, md: 0.5 } }}>
          <MetricCard {...metricCardsData[4]} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default MetricsGrid; 