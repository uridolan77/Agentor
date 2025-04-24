import { create } from 'zustand';
import dataObjectsApi, { DataModel } from '../components/data-object-canvas/data-objects-api';

interface DataModelsState {
  // Data
  dataModels: DataModel[];
  selectedDataModel: DataModel | null;
  
  // UI state
  isLoading: boolean;
  error: string | null;
  
  // Actions
  fetchDataModels: (dataSourceId?: string) => Promise<void>;
  fetchDataModel: (modelId: string) => Promise<void>;
  createDataModel: (model: Partial<DataModel>) => Promise<DataModel>;
  updateDataModel: (modelId: string, model: Partial<DataModel>) => Promise<DataModel>;
  deleteDataModel: (modelId: string) => Promise<void>;
  setSelectedDataModel: (model: DataModel | null) => void;
  clearError: () => void;
}

const useDataModelsStore = create<DataModelsState>((set, get) => ({
  // Initial state
  dataModels: [],
  selectedDataModel: null,
  isLoading: false,
  error: null,
  
  // Actions
  fetchDataModels: async (dataSourceId?: string) => {
    try {
      set({ isLoading: true, error: null });
      
      // If dataSourceId is provided, fetch models for that data source
      // Otherwise, fetch all models (we'll need to add this API endpoint)
      let models: DataModel[] = [];
      
      if (dataSourceId) {
        models = await dataObjectsApi.getDataModels(dataSourceId);
      } else {
        // For now, we'll just fetch models for all data sources
        // This is a temporary solution until we add a proper API endpoint
        const response = await fetch('/reporting/datamodels', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
          }
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch data models');
        }
        
        models = await response.json();
      }
      
      set({ dataModels: models, isLoading: false });
    } catch (error) {
      console.error('Error fetching data models:', error);
      set({ 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        isLoading: false 
      });
    }
  },
  
  fetchDataModel: async (modelId: string) => {
    try {
      set({ isLoading: true, error: null });
      const model = await dataObjectsApi.getDataModel(modelId);
      set({ selectedDataModel: model, isLoading: false });
    } catch (error) {
      console.error('Error fetching data model:', error);
      set({ 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        isLoading: false 
      });
    }
  },
  
  createDataModel: async (model: Partial<DataModel>) => {
    try {
      set({ isLoading: true, error: null });
      const newModel = await dataObjectsApi.createDataModel(model);
      
      // Update the data models list
      set(state => ({
        dataModels: [...state.dataModels, newModel],
        isLoading: false
      }));
      
      return newModel;
    } catch (error) {
      console.error('Error creating data model:', error);
      set({ 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        isLoading: false 
      });
      throw error;
    }
  },
  
  updateDataModel: async (modelId: string, model: Partial<DataModel>) => {
    try {
      set({ isLoading: true, error: null });
      const updatedModel = await dataObjectsApi.updateDataModel(modelId, model);
      
      // Update the data models list and selected model
      set(state => ({
        dataModels: state.dataModels.map(m => m.id === modelId ? updatedModel : m),
        selectedDataModel: state.selectedDataModel?.id === modelId ? updatedModel : state.selectedDataModel,
        isLoading: false
      }));
      
      return updatedModel;
    } catch (error) {
      console.error('Error updating data model:', error);
      set({ 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        isLoading: false 
      });
      throw error;
    }
  },
  
  deleteDataModel: async (modelId: string) => {
    try {
      set({ isLoading: true, error: null });
      await dataObjectsApi.deleteDataModel(modelId);
      
      // Update the data models list and clear selected model if it was deleted
      set(state => ({
        dataModels: state.dataModels.filter(m => m.id !== modelId),
        selectedDataModel: state.selectedDataModel?.id === modelId ? null : state.selectedDataModel,
        isLoading: false
      }));
    } catch (error) {
      console.error('Error deleting data model:', error);
      set({ 
        error: error instanceof Error ? error.message : 'An unknown error occurred', 
        isLoading: false 
      });
      throw error;
    }
  },
  
  setSelectedDataModel: (model: DataModel | null) => {
    set({ selectedDataModel: model });
  },
  
  clearError: () => {
    set({ error: null });
  }
}));

export default useDataModelsStore;
