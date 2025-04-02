/**
 * Utility functions for data processing and calculations
 */

// Calculate mean of array values
export const computeMean = (arr) => { 
  if (!arr || arr.length === 0) return 0;
  return arr.reduce((sum, val) => sum + parseFloat(val || 0), 0) / arr.length;
};

// Function to smooth data for better visualization
export const smoothData = (data) => {
  if (!data || data.length <= 5) return data;
  
  // For larger datasets, sample fewer points (every nth item)
  const samplingRate = Math.max(1, Math.floor(data.length / 20));
  
  // Sample data points but always include first and last
  const sampledData = [data[0]];
  
  for (let i = samplingRate; i < data.length - samplingRate; i += samplingRate) {
    sampledData.push(data[i]);
  }
  
  // Always include the last data point
  if (data.length > 1 && sampledData[sampledData.length - 1] !== data[data.length - 1]) {
    sampledData.push(data[data.length - 1]);
  }
  
  return sampledData;
};

// Function to get anomaly chip color based on score
export const getAnomalyChipColor = (score, threshold) => {
  if (!score) return 'default';
  if (score > threshold) return 'error';
  if (score > 0.75 * threshold) return 'warning';
  return 'success';
};

// Function to get anomaly status text based on score
export const getAnomalyStatus = (score, threshold) => {
  if (!score) return 'Unknown';
  if (score > threshold) return 'Suspicious';
  if (score > 0.75 * threshold) return 'Warning';
  return 'Normal';
};

// Function to get status color
export const getStatusColor = (status) => {
  switch (status.toLowerCase()) {
    case "suspicious":
      return "rgba(255, 0, 0, 1)"; // Light red
    case "warning":
      return "rgba(255, 165, 0, 1)"; // Light orange
    case "normal":
      return "rgba(41, 232, 41, 0.74)"; // Light green
    default:
      return "rgba(0, 0, 0, 0.3)"; // Default color if status is unknown
  }
}; 