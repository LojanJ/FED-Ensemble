import { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Container,
  Grid,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  LinearProgress,
  Chip,
  AppBar,
  Toolbar,
  IconButton,
  createTheme,
  ThemeProvider,
  CssBaseline,
} from '@mui/material';
import {
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  BarChart as BarChartIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Build,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PieChart, Pie, Cell } from 'recharts';
import DarkModeToggle from './Utilities';


// eslint-disable-next-line react/prop-types
const Dashboard = ({darkMode, setDarkMode}) => {

  const [metrics, setMetrics] = useState({
    num_clients: 10,
    num_rounds: 10,
    global_accuracy: 0.7,
    global_loss: 0.2,
    poison_type: 'Gradient Ascent',
    clients: [
      { id: 1, accuracy: 0.6, anomaly_score: 0.1, loss: 0.5 },
      { id: 2, accuracy: 0.7, anomaly_score: 0.20, loss: 0.5 },
      { id: 3, accuracy: 0.4, anomaly_score: 0.30, loss: 0.6 },
      { id: 4, accuracy: 0.8, anomaly_score: 0.05, loss: 0.2 },
      { id: 5, accuracy: 0.5, anomaly_score: 0.85, loss: 0.7 },
      { id: 6, accuracy: 0.9, anomaly_score: 0.03, loss: 0.1 },
      { id: 7, accuracy: 0.6, anomaly_score: 0.45, loss: 0.4 },
      { id: 8, accuracy: 0.3, anomaly_score: 0.75, loss: 0.8 },
      { id: 9, accuracy: 0.7, anomaly_score: 0.15, loss: 0.3 },
      { id: 10, accuracy: 0.6, anomaly_score: 0.55, loss: 0.5 },
    ],
    training_progress: [
      { rounds: 0, accuracy: 0.35, loss: 0.10 },
      { rounds: 1, accuracy: 0.70, loss: 0.04 },
      { rounds: 2, accuracy: 0.50, loss: 0.04 },
      { rounds: 3, accuracy: 0.40, loss: 0.04 },
      { rounds: 4, accuracy: 0.70, loss: 0.04 },
      { rounds: 5, accuracy: 0.70, loss: 0.04 },
      { rounds: 6, accuracy: 0.70, loss: 0.04 },
      { rounds: 7, accuracy: 0.70, loss: 0.04 },
      { rounds: 8, accuracy: 0.70, loss: 0.04 },
      { rounds: 9, accuracy: 0.70, loss: 0.04 }
    ]
  });

  const [config, setConfig] = useState({
    "num-server-rounds": 10,
    "fraction-fit": 0.5,
    "malicious_clients_id": [3, 7],
    "lr": 0.001,
    "input_dim": 784,
    "latent_dim": 128,
    "noise_dim": 64,
    "device": "cpu",
    "batch_size": 32
  });

  const [activeTab, setActiveTab] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchMetrics();
    // Set up an interval to fetch metrics every 10 seconds
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/metrics');
      const data = await response.json();
      setMetrics(data);
      setLastUpdated(new Date());
    } catch (error) {
        console.error("Error fetching metrics:", error);
    }
    setIsLoading(false);
  };

  // Calculate anomaly level counts
  const lowAnomalies = metrics.clients.filter(client => client.anomaly_score <= 0.4).length;
  const mediumAnomalies = metrics.clients.filter(client => client.anomaly_score > 0.4 && client.anomaly_score <= 0.7).length;
  const highAnomalies = metrics.clients.filter(client => client.anomaly_score > 0.7).length;

  const anomalyData = [
    { name: 'Low', value: lowAnomalies },
    { name: 'Medium', value: mediumAnomalies },
    { name: 'High', value: highAnomalies }
  ];

  const ANOMALY_COLORS = ['#4CAF50', '#FF9800', '#F44336'];

  // System health mock data
  const cpuUsage = 65;
  const memoryUsage = 78;

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getAnomalyChipColor = (score) => {
    if (score > 0.7) return 'error';
    if (score > 0.4) return 'warning';
    return 'success';
  };

  const getAnomalyStatus = (score) => {
    if (score > 0.7) return 'Suspicious';
    if (score > 0.4) return 'Warning';
    return 'Normal';
  };

  return (
    <Box sx={{ 
      bgcolor: 'background.default', 
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column'
    }}>

      <AppBar position="fixed" color="primary">
        <Toolbar>
          <Typography variant="body2" fontSize={{ xs: '1.1em', sm: '1.2em', md: '1.4em' }} component="div" sx={{ flexGrow: 1 }}>
            FED-Ensemble: Federated Learning System Monitor
          </Typography>
          <IconButton color='inherit' onClick={() => setDarkMode(!darkMode)}> 
            <DarkModeToggle/>
          </IconButton>
        </Toolbar>
      </AppBar>
      
      {/* Main content */}
      <Container maxWidth="lg" style={{paddingTop:'64px'}} sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        {/* Top metrics cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {/* Global Accuracy Card */}
          <Grid item xs={12} sm={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 1, 
                      bgcolor: 'primary.main', 
                      color: 'primary.contrastText',
                      mr: 2
                    }}
                  >
                    <BarChartIcon />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Centralized Global Accuracy
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="inherit" component="div">
                        {(metrics.global_accuracy * 100).toFixed(1)}%
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', ml: 1, color: 'success.main' }}>
                        {/* <TrendingUpIcon fontSize="small" />
                        <Typography variant="body2" sx={{ ml: 0.5 }}>
                          5.4%
                        </Typography> */}
                      </Box>
                    </Box>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Global Loss Card */}
          <Grid item xs={12} sm={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 1, 
                      bgcolor: 'error.main', 
                      color: 'error.contrastText',
                      mr: 2
                    }}
                  >
                    <TrendingDownIcon />
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Centralized Global Loss
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="inherit" component="div">
                        {(metrics.global_loss * 100).toFixed(1)}%
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', ml: 1, color: 'success.main' }}>
                        {/* <TrendingDownIcon fontSize="small" />
                        <Typography variant="body2" sx={{ ml: 0.5 }}>
                          2.1%
                        </Typography> */}
                      </Box>
                    </Box>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Malicious Client Ratio Card */}
          <Grid item xs={12} sm={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 1, 
                      bgcolor: 'warning.main', 
                      color: 'warning.contrastText',
                      mr: 2
                    }}
                  >
                    <WarningIcon />
                  </Box>
                  <Box>
                    <Typography variant='body2' color='text.secondary'>
                      Poisoning Method
                    </Typography>
                    <Box sx ={{display: 'flex', alignItems:'center'}}>
                      <Typography variant="inherit" component="div">
                        {(metrics.poison_type ? metrics.poison_type : 'Label Flip')}
                      </Typography>
                    </Box>
                  </Box>

                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Training Progress Chart */}
        <Card sx={{ mb: 4 }}>
          <CardHeader title="Global Model Training Progress" />
          <CardContent>

            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={metrics.training_progress}
                  margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 20,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="rounds" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#1976d2" 
                    activeDot={{ r: 8 }} 
                    name="Accuracy"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#f44336" 
                    name="Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </CardContent>
        </Card>

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
                <TableContainer component={Paper} elevation={0} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Client ID</TableCell>
                        <TableCell>Accuracy</TableCell>
                        <TableCell>Loss</TableCell>
                        <TableCell>Anomaly Score</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {metrics.clients.map((client) => (
                        <TableRow key={client.id}>
                          <TableCell>Client {client.id}</TableCell>
                          <TableCell>{(client.accuracy * 100).toFixed(1)}%</TableCell>
                          <TableCell>{client.loss.toFixed(3)}</TableCell>
                          <TableCell>
                            <Chip 
                              label={client.anomaly_score.toFixed(2)} 
                              color={getAnomalyChipColor(client.anomaly_score)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{getAnomalyStatus(client.anomaly_score)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
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
                            <Tooltip />
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
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4 }}>
                        <Box
                          sx={{
                            position: 'relative',
                            display: 'inline-flex',
                            mb: 2
                          }}
                        >
                          <Box
                            sx={{
                              width: 120,
                              height: 120,
                              borderRadius: '50%',
                              border: '12px solid',
                              borderColor: 'grey.200',
                              position: 'relative',
                              '&::before': {
                                content: '""',
                                position: 'absolute',
                                top: -12,
                                left: -12,
                                width: 'calc(100% + 24px)',
                                height: 'calc(100% + 24px)',
                                borderRadius: '50%',
                                background: `conic-gradient(${
                                  metrics.malicious_clients_ratio > 0.7
                                    ? '#f44336'
                                    : metrics.malicious_clients_ratio > 0.3
                                    ? '#ff9800'
                                    : '#4caf50'
                                } ${metrics.malicious_clients_ratio * 360}deg, transparent 0)`,
                                transform: 'rotate(-90deg)',
                              }
                            }}
                          />
                          <Box
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              bottom: 0,
                              right: 0,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Typography variant="h5" component="div" fontWeight="bold">
                              {(metrics.malicious_clients_ratio * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        </Box>
                        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-around', width: '100%' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: 12, height: 12, bgcolor: 'success.main', borderRadius: '50%', mr: 1 }} />
                            <Typography variant="caption">Safe (0-30%)</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: 12, height: 12, bgcolor: 'warning.main', borderRadius: '50%', mr: 1 }} />
                            <Typography variant="caption">Warning (30-70%)</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: 12, height: 12, bgcolor: 'error.main', borderRadius: '50%', mr: 1 }} />
                            <Typography variant="caption">Critical (70%+)</Typography>
                          </Box>
                        </Box>
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
            <Card>
              <Tab label="System Performance" icon={<Build />} iconPosition="start" />
              <CardContent>
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">CPU Usage</Typography>
                    <Typography variant="body2">{cpuUsage}%</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={cpuUsage} 
                    color={cpuUsage > 80 ? 'error' : cpuUsage > 60 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Memory Usage</Typography>
                    <Typography variant="body2">{memoryUsage}%</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={memoryUsage}
                    color={memoryUsage > 80 ? 'error' : memoryUsage > 60 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Model Configuration */}
          <Grid item xs={12} md={6}>
            <Card>
            <Tab label="System Configuration" icon={<SettingsIcon />} iconPosition="start" />
              <CardContent>
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
                              {Array.isArray(value) ? value.join(', ') : JSON.stringify(value)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Footer */}
      <Box 
        component="footer" 
        sx={{ 
          py: 3, 
          px: 2, 
          mt: 'auto', 
          bgcolor: 'background.paper', 
          borderTop: 1, 
          borderColor: 'divider'
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            Last updated: {lastUpdated.toLocaleString()} 
            {isLoading && <Box component="span" sx={{ ml: 1, display: 'inline-block', animation: 'spin 1s linear infinite' }}>⟳</Box>}
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};


const App = () => {
  const [darkMode, setDarkMode] = useState(false);
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light'
    }
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline/>
      <Dashboard darkMode={darkMode} setDarkMode={setDarkMode}/>
    </ThemeProvider>
  )
}
export default App;    