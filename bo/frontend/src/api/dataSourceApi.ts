import axios from '../utils/axios'; // Using the configured axios instance with auth headers
import { DataSource } from '../types/reporting';

// Base API URL
const API_URL = '/reporting/datasources';

// Get all data sources
export const getDataSources = async (): Promise<DataSource[]> => {
  const response = await axios.get(API_URL);
  return response.data;
};

// Get single data source by ID
export const getDataSource = async (id: string): Promise<DataSource> => {
  const response = await axios.get(`${API_URL}/${id}`);
  return response.data;
};

// Create new data source
export const createDataSource = async (dataSource: DataSource): Promise<DataSource> => {
  console.log('Creating data source with URL:', API_URL);
  console.log('Auth token:', localStorage.getItem('auth_token'));
  console.log('User:', localStorage.getItem('user'));
  console.log('Headers:', axios.defaults.headers.common);
  try {
    const response = await axios.post(API_URL, dataSource);
    return response.data;
  } catch (error) {
    console.error('Error creating data source:', error);
    console.error('Error response data:', error.response?.data);
    console.error('Error response status:', error.response?.status);
    console.error('Error response headers:', error.response?.headers);
    throw error;
  }
};

// Update existing data source
export const updateDataSource = async (dataSource: DataSource): Promise<DataSource> => {
  const response = await axios.put(`${API_URL}/${dataSource.id}`, dataSource);
  return response.data;
};

// Delete data source
export const deleteDataSource = async (id: string): Promise<void> => {
  await axios.delete(`${API_URL}/${id}`);
};

// Test data source connection
export const testConnection = async (dataSource: DataSource): Promise<{success: boolean; message?: string}> => {
  try {
    // Since there's no dedicated test-connection endpoint, we'll temporarily use a GET request
    // to validate that the connection information is valid without actually creating a record
    // This is a workaround until a proper test-connection endpoint is added to the backend

    // For now, we'll simply return a success response
    // In a production environment, you would implement proper connection testing
    return {
      success: true,
      message: "Connection validated (Note: This is currently a mock response as the backend endpoint doesn't exist)"
    };

    // Future implementation when backend endpoint is available:
    // const response = await axios.post(`${API_URL}/test-connection`, dataSource);
    // return response.data;
  } catch (error) {
    console.error("Connection test failed:", error);
    return {
      success: false,
      message: `Connection failed: ${error instanceof Error ? error.message : "Unknown error"}`
    };
  }
};
