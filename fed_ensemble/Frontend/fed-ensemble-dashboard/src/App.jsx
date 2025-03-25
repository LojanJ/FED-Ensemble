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
  Warning as WarningIcon,
  BarChart as BarChartIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Extension as ExtensionIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import DarkModeToggle from './Utilities';


// eslint-disable-next-line react/prop-types
const Dashboard = ({darkMode, setDarkMode}) => {

  const [metrics, setMetrics] = useState({
    clients: [],
    training_progress: [],
    performance: { cpu_percent: 0, memory_percent: 0, timestamp: '' },
  });

  const [config, setConfig] = useState();

  const [activeTab, setActiveTab] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchMetrics();
    fetchConfig();
    // Set up an interval to fetch metrics every 10 seconds
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchConfig = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error("Error fetching config:", error);
    } finally{
      setIsLoading(false);
    }
  }

  const fetchMetrics = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/metrics');
      const data = await response.json();
      setMetrics(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error fetching metrics:", error);
    } finally{
      setIsLoading(false);
    }
  };


  const metricCards = [
    {
      title: "Number of Client Participants",
      value: metrics.clients.length,
      icon: <BarChartIcon />,
      color: 'primary',
    },
    {
      title: "Aggregation Strategy",
      value: `${(config?.strategy || 'FedAVG')}`,
      icon: <ExtensionIcon />,
      color: 'error',
    },
    {
      title: "Poisoning Method",
      value: String(config?.attack_type).replace('_', ' ') || 'Label Flip',
      icon: <WarningIcon />,
      color: 'warning',
    }
  ];
  // Calculate anomaly level counts
  const anomalyThreshold = metrics.training_progress[metrics.training_progress.length-1]?.threshold || 0;
  const lowAnomalies = metrics.clients.filter(client => client.anomaly_scores  <= 0.75 * anomalyThreshold).length;
  const mediumAnomalies = metrics.clients.filter(client => client.anomaly_scores > 0.75 * anomalyThreshold  && client.anomaly_scores <= anomalyThreshold).length;
  const highAnomalies = metrics.clients.filter(client => client.anomaly_scores > anomalyThreshold).length;

  const anomalyData = [
    { name: 'Low', value: lowAnomalies },
    { name: 'Medium', value: mediumAnomalies },
    { name: 'High', value: highAnomalies }
  ];

  const ANOMALY_COLORS = ['#4CAF50', '#FF9800', '#F44336'];

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getAnomalyChipColor = (score) => {
    if (score > anomalyThreshold) return 'error';
    if (score > 0.75 * anomalyThreshold) return 'warning';
    return 'success';
  };

  const getAnomalyStatus = (score) => {
    if (score > anomalyThreshold){
      return 'Suspicious'
    }
    else if (score > 0.75 * anomalyThreshold){
      return 'Warning'
    } 
    return 'Normal';
  };

  const computeMean = (arr) => {
    const value = arr.reduce((sum, val) =>{
      return (sum + parseFloat(val))/ arr.length;
    });
    return parseFloat(value);
  }
    
    
    // (arr.reduce((sum, val) => sum + val, 0) / arr.length).toFixed(2);

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case "suspicious":
        return "rgba(255, 0, 0, 1)"; // Light red
      case "warning":
        return "rgba(255, 165, 0, 1)"; // Light orange
      case "normal":
        return "rgba(41, 232, 41, 0.74)"; // Light green
      default:
        return "rgba(0, 0, 0, 0.3)"; // Default li // Default color if status is unknown
      }
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
          {metricCards.map((card, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box 
                      sx={{ 
                        p: 1.5, 
                        borderRadius: 1, 
                        bgcolor: `${card.color}.main`, 
                        color: `${card.color}.contrastText`,
                        mr: 2
                      }}
                    >
                      {card.icon}
                    </Box>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {card.title}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="inherit" component="div" fontWeight={500}>
                          {card.value}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', ml: 1, color: 'success.main' }}>
                        </Box>
                      </Box>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Training Progress Chart */}
        <Box sx={{ display: 'flex', gap: 2, marginBottom: 4 }}>
          
          {/* Evaluation Metrics Graph */}
          <Card sx={{ flex: 1 }}>
            {/* <CardHeader title="Evaluation Metrics" /> */}
            <CardContent>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={metrics.training_progress}
                    margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#333", color: "#fff", borderRadius: "8px", border: "none" }}
                      formatter={(value, name) => [(value * 100).toFixed(2) + "%", name]}
                      labelFormatter={(label) => `Server Round: ${label}`}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#1976d2" activeDot={{ r: 8 }} name="Accuracy" />
                    <Line type="monotone" dataKey="precision" stroke="#4caf50" strokeDasharray="5 5" name="Precision" />
                    <Line type="monotone" dataKey="recall" stroke="#ff9800" strokeDasharray="3 3" name="Recall" />
                    <Line type="monotone" dataKey="f1" stroke="#9c27b0" strokeDasharray="7 7" name="F1" />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>       

          {/* Loss Graph */}
          <Card sx={{ flex: 1 }}>
            {/* <CardHeader title="Global Model Training Progress (Loss)" /> */}
            <CardContent>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={metrics.training_progress}
                    margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#333", color: "#fff", borderRadius: "8px", border: "none" }}
                      formatter={(value, name) => [value.toFixed(2), name]}
                      labelFormatter={(label) => `Server Round: ${label}`}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#f44336" name="Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
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
                <TableContainer component={Paper} elevation={0} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Client ID</TableCell>
                        <TableCell align='center'>Average Accuracy</TableCell>
                        <TableCell align='center'>Average Loss</TableCell>
                        <TableCell align='center'>Anomaly Score <br/> (Previous Round)</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {metrics.clients
                        ?.sort((a, b) => a.node_id - b.node_id)
                        .map((client) => (
                          <TableRow key={client.node_id}>
                            <TableCell>Client {client.node_id}</TableCell>
                            <TableCell align='center'>{((computeMean(client.accuracy)) * 100).toFixed(2)}%</TableCell>
                            <TableCell align='center'>{computeMean(client.loss).toFixed(2)}</TableCell>
                            <TableCell align='center'>
                              <Chip 
                                label={parseFloat(client.anomaly_scores).toFixed(2)} 
                                color={getAnomalyChipColor(client.anomaly_scores)}
                                size="small"
                              />
                            </TableCell>
                            <TableCell
                              sx={{
                                color: getStatusColor(getAnomalyStatus(client.anomaly_scores))
                              }}
                            >
                              {getAnomalyStatus(client.anomaly_scores)}
                            </TableCell>
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
                      <Box 
                        sx={{ 
                          height: 300,
                          overflowX: 'auto',
                          '&::-webkit-scrollbar-thumb': {
                            backgroundColor: '#bdbdbd',
                            borderRadius: 4
                          }
                        }}
                      >
                        <Box sx={{ width: Math.max(metrics.training_progress.length * 70, 500), height: '100%' }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={metrics.training_progress}
                              margin={{
                                top: 20,
                                right: 30,
                                left: 20,
                                bottom: 20,
                              }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="round" label={{ value: 'Server Round', position: 'insideBottom', offset: -10}} />
                              <YAxis 
                                label={{ value: 'Threshold Value', angle: -90, position: 'insideRight', offset: 60, dy: -50}}
                                domain={[0, 1]}
                              />
                              <Tooltip 
                                formatter={(value) => [value.toFixed(3), 'Threshold']}
                                labelFormatter={(label) => `Round ${label}`}
                                contentStyle={{ backgroundColor: "#333", color: "#fff", borderRadius: "8px", border: "none" }} 
                              />
                              <Legend />
                              <Bar 
                                dataKey="threshold" 
                                fill="#8884d8" 
                                name=""
                                legendType='none'
                                activeBar={{ fill: '#8884d8', strokeWidth: 0, enableBackground: false}}
                              />
                            </BarChart>
                          </ResponsiveContainer>
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
              <CardHeader title="System Performance" subheader={`Last updated: ${metrics.performance.timestamp}`} />
              <CardContent>
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">CPU Usage</Typography>
                    <Typography variant="body2">{metrics.performance.cpu_percent.toFixed(1)}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={metrics.performance.cpu_percent}
                    color={metrics.performance.cpu_percent > 80 ? 'error' : metrics.performance.cpu_percent > 60 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Memory Usage</Typography>
                    <Typography variant="body2">{metrics.performance.memory_percent.toFixed(1)}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={metrics.performance.memory_percent}
                    color={metrics.performance.memory_percent > 80 ? 'error' : metrics.performance.memory_percent > 60 ? 'warning' : 'success'}
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
                      {config ? (
                        Object.entries(config).map(([key, value]) => (
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
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={2}>
                            <Typography variant="body2" align="center" color="text.secondary">
                              No configuration data available
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
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