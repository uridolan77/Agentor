import axios from 'axios';

// Types for the API calls
export interface TableSchema {
  name: string;
  columns: ColumnSchema[];
}

export interface ColumnSchema {
  name: string;
  type: string;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
  references?: {
    table: string;
    column: string;
  };
}

export interface Relationship {
  id: string;
  sourceTable: string;
  sourceColumn: string;
  targetTable: string;
  targetColumn: string;
  type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
}

export interface DataModel {
  id: string;
  name: string;
  dataSourceId: string;
  tables: string[];
  relationships: Relationship[];
  layout?: string; // JSON string containing layout information
  createdAt: string;
  updatedAt: string;
}

// Base API URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:9000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout to prevent infinite hanging requests
  timeout: 30000
});

// Add authorization interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Custom error handling and transformation
const handleApiError = (error: any, customMessage?: string) => {
  // Standardize the error format
  let errorMessage = customMessage || 'An error occurred';
  let errorDetails = '';

  if (axios.isAxiosError(error)) {
    // Handle Axios specific errors
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      errorMessage = error.response.data?.message || error.message;
      errorDetails = error.response.data?.detail || `Status code: ${error.response.status}`;

      // Special handling for specific status codes
      if (error.response.status === 404) {
        errorMessage = 'Resource not found';
        errorDetails = 'The requested endpoint does not exist or is not configured correctly.';
      } else if (error.response.status === 401) {
        errorMessage = 'Authentication required';
        errorDetails = 'Please log in again to continue.';
      } else if (error.response.status === 403) {
        errorMessage = 'Access forbidden';
        errorDetails = 'You do not have permission to access this resource.';
      } else if (error.response.status === 500) {
        errorMessage = 'Server error';
        errorDetails = 'The server encountered an error. Please try again later.';
      }
    } else if (error.request) {
      // The request was made but no response was received
      errorMessage = 'No response from server';
      errorDetails = 'Please check your network connection and try again.';
    } else {
      // Something happened in setting up the request
      errorMessage = error.message || 'Request configuration error';
    }
  } else if (error instanceof Error) {
    errorMessage = error.message;
  }

  // Create a standardized error object
  const enhancedError = new Error(errorMessage);
  (enhancedError as any).details = errorDetails;
  (enhancedError as any).originalError = error;

  console.error(`API Error: ${errorMessage}`, { details: errorDetails, originalError: error });

  throw enhancedError;
};

// API functions for data objects

/**
 * Get database schema (tables and columns) for a data source
 */
export const getDatabaseSchema = async (dataSourceId: string): Promise<TableSchema[]> => {
  try {
    const response = await api.get(`/reporting/datasources/${dataSourceId}/schema`);
    return response.data;
  } catch (error) {
    return handleApiError(error, 'Failed to fetch database schema');
  }
};

/**
 * Get table columns for a specific table in a data source
 */
export const getTableColumns = async (dataSourceId: string, tableName: string): Promise<ColumnSchema[]> => {
  try {
    const response = await api.get(`/reporting/datasources/${dataSourceId}/tables/${tableName}/columns`);
    return response.data;
  } catch (error) {
    return handleApiError(error, `Failed to fetch columns for table ${tableName}`);
  }
};

/**
 * Create a new data model
 */
export const createDataModel = async (dataModel: Partial<DataModel>): Promise<DataModel> => {
  try {
    const response = await api.post('/reporting/datamodels', dataModel);
    return response.data;
  } catch (error) {
    return handleApiError(error, 'Error creating data model');
  }
};

/**
 * Update an existing data model
 */
export const updateDataModel = async (modelId: string, dataModel: Partial<DataModel>): Promise<DataModel> => {
  try {
    const response = await api.put(`/reporting/datamodels/${modelId}`, dataModel);
    return response.data;
  } catch (error) {
    return handleApiError(error, `Error updating data model ${modelId}`);
  }
};

/**
 * Get a list of all data models for a data source
 */
export const getDataModels = async (dataSourceId: string): Promise<DataModel[]> => {
  try {
    const response = await api.get(`/reporting/datamodels?dataSourceId=${dataSourceId}`);
    return response.data;
  } catch (error) {
    return handleApiError(error, 'Error fetching data models');
  }
};

/**
 * Get a specific data model by ID
 */
export const getDataModel = async (modelId: string): Promise<DataModel> => {
  try {
    const response = await api.get(`/reporting/datamodels/${modelId}`);
    return response.data;
  } catch (error) {
    return handleApiError(error, `Error fetching data model ${modelId}`);
  }
};

/**
 * Delete a data model
 */
export const deleteDataModel = async (modelId: string): Promise<void> => {
  try {
    await api.delete(`/reporting/datamodels/${modelId}`);
  } catch (error) {
    return handleApiError(error, `Error deleting data model ${modelId}`);
  }
};

/**
 * Generate SQL from a data model
 */
export const generateSqlFromModel = async (modelId: string, options: {
  dialect?: 'mysql' | 'postgresql' | 'sqlite';
  includeDropTable?: boolean;
  includeForeignKeys?: boolean;
}): Promise<string> => {
  try {
    const response = await api.post(`/reporting/datamodels/${modelId}/generate-sql`, options);
    return response.data.sql;
  } catch (error) {
    return handleApiError(error, `Error generating SQL for model ${modelId}`);
  }
};

/**
 * Test a query on a data source
 */
export const testQuery = async (dataSourceId: string, query: string): Promise<any> => {
  try {
    const response = await api.post(`/reporting/datasources/${dataSourceId}/test-query`, { query });
    return response.data;
  } catch (error) {
    return handleApiError(error, 'Error testing query');
  }
};

export default {
  getDatabaseSchema,
  getTableColumns,
  createDataModel,
  updateDataModel,
  getDataModels,
  getDataModel,
  deleteDataModel,
  generateSqlFromModel,
  testQuery
};
