/* eslint-disable react/prop-types */
import { Card, CardHeader, CardContent, Box, Typography, LinearProgress } from '@mui/material';

const SystemHealthPanel = ({ performance }) => {
  return (
    <Card>
      <CardHeader title="System Performance" subheader={`Last updated: ${performance.timestamp}`} />
      <CardContent>
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">CPU Usage</Typography>
            <Typography variant="body2">{performance.cpu_percent.toFixed(1)}%</Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={performance.cpu_percent}
            color={performance.cpu_percent > 80 ? 'error' : performance.cpu_percent > 60 ? 'warning' : 'success'}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">Memory Usage</Typography>
            <Typography variant="body2">{performance.memory_percent.toFixed(1)}%</Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={performance.memory_percent}
            color={performance.memory_percent > 80 ? 'error' : performance.memory_percent > 60 ? 'warning' : 'success'}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default SystemHealthPanel; 