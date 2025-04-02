/**
 * API service functions for fetching data from the backend
 */

const API_BASE_URL = 'http://localhost:5000';

// Fetch metrics data
export const fetchMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching metrics:", error);
    throw error;
  }
};

// Fetch configuration data - fix URL if needed
export const fetchConfig = async () => {
  try {
    // Try the standard endpoint first
    const response = await fetch(`${API_BASE_URL}/config`);
    if (!response.ok) {
      // If that fails, try alternative endpoints
      console.log("Trying alternative config endpoint...");
      const altResponse = await fetch(`${API_BASE_URL}/configuration`);
      if (!altResponse.ok) {
        throw new Error(`HTTP error! Status: ${altResponse.status}`);
      }
      return await altResponse.json();
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching config:", error);
    throw error;
  }
}; 