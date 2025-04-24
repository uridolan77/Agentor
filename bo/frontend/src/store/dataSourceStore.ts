import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import * as dataSourceApi from '../api/dataSourceApi';
import { DataSource } from '../types/reporting';
import { 
  adaptApiResponse, 
  adaptApiResponseArray, 
  snakeToCamelDataSource, 
  camelToSnakeDataSource 
} from '../utils/typeAdapters';

interface ConnectionTestStatus {
  tested: boolean;
  success: boolean;
  message: string;
}

interface DataSourceState {
  // Data
  dataSources: DataSource[];
  selectedDataSource: DataSource | null;
  loading: boolean;
  error: string | null;

  // Connection testing
  connectionTestStatus: ConnectionTestStatus;

  // Actions
  fetchDataSources: () => Promise<void>;
  createDataSource: (dataSource: DataSource) => Promise<DataSource>;
  updateDataSource: (dataSource: DataSource) => Promise<DataSource>;
  deleteDataSource: (dataSourceId: string) => Promise<void>;
  testConnection: (dataSource: DataSource) => Promise<void>;
  resetConnectionTest: () => void;
  selectDataSource: (dataSourceId: string | null) => void;
}

const initialConnectionTestStatus: ConnectionTestStatus = {
  tested: false,
  success: false,
  message: '',
};

const useDataSourceStore = create<DataSourceState>((set, get) => ({
  // Initial state
  dataSources: [],
  selectedDataSource: null,
  loading: false,
  error: null,
  connectionTestStatus: initialConnectionTestStatus,

  // Actions
  fetchDataSources: async () => {
    set({ loading: true, error: null });
    try {
      const apiDataSources = await dataSourceApi.getDataSources();
      // Convert API response to the expected format
      const dataSources = adaptApiResponseArray(apiDataSources, snakeToCamelDataSource);
      set({ dataSources, loading: false });
    } catch (error) {
      console.error('Error fetching data sources:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch data sources',
        loading: false
      });
    }
  },

  createDataSource: async (dataSource: DataSource) => {
    set({ loading: true, error: null });
    try {
      // Convert to API format before sending
      const apiDataSource = camelToSnakeDataSource(dataSource);
      const createdApiDataSource = await dataSourceApi.createDataSource(apiDataSource);
      // Convert API response back to the expected format
      const createdDataSource = snakeToCamelDataSource(createdApiDataSource);
      
      set((state) => ({
        dataSources: [...state.dataSources, createdDataSource],
        loading: false
      }));
      return createdDataSource;
    } catch (error) {
      console.error('Error creating data source:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to create data source',
        loading: false
      });
      throw error;
    }
  },

  updateDataSource: async (dataSource: DataSource) => {
    set({ loading: true, error: null });
    try {
      // Convert to API format before sending
      const apiDataSource = camelToSnakeDataSource(dataSource);
      const updatedApiDataSource = await dataSourceApi.updateDataSource(apiDataSource);
      // Convert API response back to the expected format
      const updatedDataSource = snakeToCamelDataSource(updatedApiDataSource);
      
      set((state) => ({
        dataSources: state.dataSources.map(ds =>
          ds.id === updatedDataSource.id ? updatedDataSource : ds
        ),
        selectedDataSource: state.selectedDataSource?.id === updatedDataSource.id
          ? updatedDataSource
          : state.selectedDataSource,
        loading: false
      }));
      return updatedDataSource;
    } catch (error) {
      console.error('Error updating data source:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to update data source',
        loading: false
      });
      throw error;
    }
  },

  deleteDataSource: async (dataSourceId: string) => {
    set({ loading: true, error: null });
    try {
      await dataSourceApi.deleteDataSource(dataSourceId);
      set((state) => ({
        dataSources: state.dataSources.filter(ds => ds.id !== dataSourceId),
        selectedDataSource: state.selectedDataSource?.id === dataSourceId
          ? null
          : state.selectedDataSource,
        loading: false
      }));
    } catch (error) {
      console.error('Error deleting data source:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to delete data source',
        loading: false
      });
      throw error;
    }
  },

  testConnection: async (dataSource: DataSource) => {
    set({
      connectionTestStatus: {
        ...initialConnectionTestStatus,
        tested: false
      },
      loading: true,
      error: null
    });

    try {
      // Convert to API format before sending
      const apiDataSource = camelToSnakeDataSource(dataSource);
      const result = await dataSourceApi.testConnection(apiDataSource);
      set({
        connectionTestStatus: {
          tested: true,
          success: result.success,
          message: result.message || ''
        },
        loading: false
      });
    } catch (error) {
      console.error('Error testing connection:', error);
      set({
        connectionTestStatus: {
          tested: true,
          success: false,
          message: error instanceof Error ? error.message : 'Connection test failed'
        },
        loading: false
      });
    }
  },

  resetConnectionTest: () => {
    set({ connectionTestStatus: initialConnectionTestStatus });
  },

  selectDataSource: (dataSourceId: string | null) => {
    if (!dataSourceId) {
      set({ selectedDataSource: null });
      return;
    }

    const { dataSources } = get();
    const dataSource = dataSources.find(ds => ds.id === dataSourceId) || null;
    set({ selectedDataSource: dataSource });
  }
}));

export default useDataSourceStore;
