import { useState, useEffect } from 'react';
import { fetchMetrics, fetchConfig } from '../utils/api';
import { smoothData } from '../utils/dataHelpers';

export const useDataFetching = () => {
  const [metrics, setMetrics] = useState({
    clients: [],
    training_progress: [],
    performance: { cpu_percent: 0, memory_percent: 0, timestamp: '' },
  });
  
  const [config, setConfig] = useState();
  const [processedData, setProcessedData] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // 'active', 'connecting', 'error'

  // Process metrics data for visualization
  useEffect(() => {
    if (metrics.training_progress && metrics.training_progress.length > 0) {
      // If we have too many data points, reduce them to improve chart readability
      let processedMetrics;
      
      if (metrics.training_progress.length > 20) {
        // Apply smoothing for large datasets
        processedMetrics = smoothData(metrics.training_progress);
      } else {
        processedMetrics = metrics.training_progress;
      }
      
      setProcessedData(processedMetrics);
    }
  }, [metrics]);

  // Function to fetch metrics data
  const getMetrics = async () => {
    setIsLoading(true);
    try {
      const data = await fetchMetrics();
      setMetrics(data);
      setLastUpdated(new Date());
      setConnectionStatus('active');
    } catch (error) {
      console.error("Error fetching metrics:", error);
      setConnectionStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to fetch configuration data
  const getConfig = async () => {
    setIsLoading(true);
    try {
      const data = await fetchConfig();
      setConfig(data);
      setConnectionStatus('active');
    } catch (error) {
      console.error("Error fetching config:", error);
      setConnectionStatus('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Initial data fetching
  useEffect(() => {
    getMetrics();
    getConfig();
    
    // Set up an interval to fetch metrics every 10 seconds
    const interval = setInterval(getMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  // Add a reconnection attempt when in error state
  useEffect(() => {
    let reconnectTimer;
    
    if (connectionStatus === 'error') {
      // Try reconnecting every 5 seconds if connection is lost
      reconnectTimer = setTimeout(() => {
        console.log('Attempting to reconnect...');
        getMetrics();
        getConfig();
      }, 5000);
    }
    
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [connectionStatus]);

  return {
    metrics,
    config, 
    processedData,
    lastUpdated,
    isLoading,
    connectionStatus,
    refreshData: () => {
      getMetrics();
      getConfig();
    }
  };
}; 