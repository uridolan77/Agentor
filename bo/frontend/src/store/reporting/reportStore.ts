import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import reportingAPI from '../../api/reportingApi';
import {
  Report,
  DataSource,
  Dimension,
  Metric,
  CalculatedMetric,
  ReportData,
  ReportCreate,
  ReportUpdate,
  ReportElement,
  Visualization,
  VisualizationType,
  FilterCondition,
  SortCondition
} from '../../types/reporting/reportTypes';
// Import the adapter functions but don't use them directly with the imported types
// Instead, use type assertions to make TypeScript happy
import {
  adaptApiResponse,
  adaptApiResponseArray
} from '../../utils/typeAdapters';

// Reports List Store
interface ReportsListState {
  reports: Report[];
  isLoading: boolean;
  error: string | null;
  loadReports: (filters?: Record<string, any>) => Promise<void>;
  deleteReport: (id: string) => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  clearError: () => void;
}

export const useReportsListStore = create<ReportsListState>()(
  devtools(
    (set, get) => ({
      reports: [],
      isLoading: false,
      error: null,

      loadReports: async (filters?: Record<string, any>) => {
        set({ isLoading: true, error: null });
        try {
          // The reports API endpoint is not implemented yet, so return an empty array
          // const reports = await reportingAPI.getReports(filters);
          const reports: Report[] = [];
          set({ reports, isLoading: false });
        } catch (error) {
          console.error('Error loading reports:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to load reports',
            isLoading: false
          });
        }
      },

      deleteReport: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          await reportingAPI.deleteReport(id);
          set(state => ({
            reports: state.reports.filter(report => report.id !== id),
            isLoading: false
          }));
        } catch (error) {
          console.error('Error deleting report:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to delete report',
            isLoading: false
          });
        }
      },

      toggleFavorite: async (id: string) => {
        try {
          // Find the current report to get its favorite status
          const report = get().reports.find(r => r.id === id);
          if (!report) {
            throw new Error('Report not found');
          }

          // Toggle the favorite status by passing the opposite of the current value
          const { isFavorite } = await reportingAPI.toggleFavoriteReport(id, !report.isFavorite);
          set(state => ({
            reports: state.reports.map(report =>
              report.id === id ? { ...report, isFavorite } : report
            )
          }));
        } catch (error) {
          console.error('Error toggling favorite:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to update favorite status'
          });
        }
      },

      clearError: () => set({ error: null })
    }),
    { name: 'reports-list-store' }
  )
);

// Data Sources Store
interface DataSourcesState {
  dataSources: DataSource[];
  isLoading: boolean;
  error: string | null;
  loadDataSources: () => Promise<void>;
  createDataSource: (dataSource: any) => Promise<DataSource>;
  updateDataSource: (id: string, dataSource: any) => Promise<DataSource>;
  deleteDataSource: (id: string) => Promise<void>;
  clearError: () => void;
}

export const useDataSourcesStore = create<DataSourcesState>()(
  devtools(
    (set, get) => ({
      dataSources: [],
      isLoading: false,
      error: null,

      loadDataSources: async () => {
        set({ isLoading: true, error: null });
        try {
          const dataSources = await reportingAPI.getDataSources();
          // No need for conversion since the API already returns the expected format
          set({ dataSources, isLoading: false });
        } catch (error) {
          console.error('Error loading data sources:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to load data sources',
            isLoading: false
          });
        }
      },

      createDataSource: async (dataSource: any) => {
        set({ isLoading: true, error: null });
        try {
          // Use the dataSource directly
          const newDataSource = await reportingAPI.createDataSource(dataSource);

          set(state => ({
            dataSources: [...state.dataSources, newDataSource],
            isLoading: false
          }));
          return newDataSource;
        } catch (error) {
          console.error('Error creating data source:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to create data source',
            isLoading: false
          });
          throw error;
        }
      },

      updateDataSource: async (id: string, dataSource: any) => {
        set({ isLoading: true, error: null });
        try {
          // Use the dataSource directly
          const updatedDataSource = await reportingAPI.updateDataSource(id, dataSource);

          set(state => ({
            dataSources: state.dataSources.map(ds =>
              ds.id === id ? updatedDataSource : ds
            ),
            isLoading: false
          }));
          return updatedDataSource;
        } catch (error) {
          console.error('Error updating data source:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to update data source',
            isLoading: false
          });
          throw error;
        }
      },

      deleteDataSource: async (id: string) => {
        set({ isLoading: true, error: null });
        try {
          await reportingAPI.deleteDataSource(id);
          set(state => ({
            dataSources: state.dataSources.filter(ds => ds.id !== id),
            isLoading: false
          }));
        } catch (error) {
          console.error('Error deleting data source:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to delete data source',
            isLoading: false
          });
          throw error;
        }
      },

      clearError: () => set({ error: null })
    }),
    { name: 'data-sources-store' }
  )
);

// Report Builder Store
interface ReportBuilderState {
  report: Report;
  dataSource: DataSource | null;
  availableFields: {
    dimensions: Dimension[];
    metrics: Metric[];
    calculatedMetrics: CalculatedMetric[];
  };
  previewData: ReportData | null;
  isLoading: boolean;
  error: string | null;
  isDirty: boolean;

  createNewReport: (dataSourceId: string) => void;
  loadReport: (reportId: string) => Promise<void>;
  saveReport: (isPublic?: boolean) => Promise<string>;
  updateReportName: (name: string) => void;
  updateReportDescription: (description: string) => void;
  loadDataSource: (dataSourceId: string) => Promise<void>;
  loadAvailableFields: (dataSourceId: string) => Promise<void>;
  generatePreview: () => Promise<void>;
  addElement: (element: Partial<ReportElement>) => void;
  updateElement: (elementId: string, updates: Partial<ReportElement>) => void;
  removeElement: (elementId: string) => void;
  selectCanvasElement: (elementId?: string) => void;
  clearError: () => void;
}

// Create a default empty report
const createEmptyReport = (): Report => ({
  id: '',
  name: 'New Report',
  description: '',
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  createdBy: '',
  dataSourceId: '',
  isPublic: false,
  isFavorite: false,
  elements: []
});

export const useReportBuilderStore = create<ReportBuilderState>()(
  devtools(
    (set, get) => ({
      report: createEmptyReport(),
      dataSource: null,
      availableFields: {
        dimensions: [],
        metrics: [],
        calculatedMetrics: []
      },
      previewData: null,
      isLoading: false,
      error: null,
      isDirty: false,

      createNewReport: (dataSourceId: string) => {
        const newReport = createEmptyReport();
        newReport.dataSourceId = dataSourceId;

        set({
          report: newReport,
          isDirty: true
        });

        // Load data source and available fields
        get().loadDataSource(dataSourceId);
        get().loadAvailableFields(dataSourceId);
      },

      loadReport: async (reportId: string) => {
        set({ isLoading: true, error: null });
        try {
          const report = await reportingAPI.getReport(reportId);
          // No need for conversion since the API already returns the expected format

          set({ report, isLoading: false, isDirty: false });

          // Load data source and available fields
          await get().loadDataSource(report.dataSourceId);
          await get().loadAvailableFields(report.dataSourceId);

          // Generate preview data
          await get().generatePreview();
        } catch (error) {
          console.error('Error loading report:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to load report',
            isLoading: false
          });
          throw error;
        }
      },

      saveReport: async (isPublic = false) => {
        set({ isLoading: true, error: null });
        try {
          const { report } = get();

          // Update isPublic flag
          const reportData = {
            ...report,
            isPublic
          };

          let savedReport: Report;

          if (report.id) {
            // Update existing report
            savedReport = await reportingAPI.updateReport(report.id, reportData);
          } else {
            // Create new report
            savedReport = await reportingAPI.createReport(reportData as ReportCreate);
          }

          set({
            report: savedReport,
            isLoading: false,
            isDirty: false
          });

          return savedReport.id;
        } catch (error) {
          console.error('Error saving report:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to save report',
            isLoading: false
          });
          throw error;
        }
      },

      updateReportName: (name: string) => {
        set(state => ({
          report: { ...state.report, name },
          isDirty: true
        }));
      },

      updateReportDescription: (description: string) => {
        set(state => ({
          report: { ...state.report, description },
          isDirty: true
        }));
      },

      loadDataSource: async (dataSourceId: string) => {
        set({ isLoading: true, error: null });
        try {
          const dataSource = await reportingAPI.getDataSource(dataSourceId);
          // No need for conversion since the API already returns the expected format
          set({ dataSource, isLoading: false });
        } catch (error) {
          console.error('Error loading data source:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to load data source',
            isLoading: false
          });
        }
      },

      loadAvailableFields: async (dataSourceId: string) => {
        set({ isLoading: true, error: null });
        try {
          const fields = await reportingAPI.getDataSourceFields(dataSourceId);
          set({ availableFields: fields, isLoading: false });
        } catch (error) {
          console.error('Error loading fields:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to load fields',
            isLoading: false
          });
        }
      },

      generatePreview: async () => {
        set({ isLoading: true, error: null });
        try {
          const { report } = get();

          // Only generate preview if report has elements
          if (report.elements.length === 0) {
            set({ previewData: null, isLoading: false });
            return;
          }

          const previewData = await reportingAPI.generateReportData({
            reportId: report.id || 'preview',
            parameters: report.parameters
          });

          set({ previewData, isLoading: false });
        } catch (error) {
          console.error('Error generating preview:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to generate preview',
            isLoading: false
          });
        }
      },

      addElement: (element: Partial<ReportElement>) => {
        const { report } = get();

        // Create a new element with defaults
        const newElement: ReportElement = {
          id: `temp-${Date.now()}`,
          name: element.name || 'New Element',
          reportId: report.id,
          dimensions: element.dimensions || [],
          metrics: element.metrics || [],
          filters: element.filters || [],
          sort: element.sort || [],
          visualization: element.visualization || {
            type: VisualizationType.TABLE,
            title: 'New Element'
          },
          position: element.position || {
            x: 0,
            y: 0,
            width: 6,
            height: 4
          }
        };

        set(state => ({
          report: {
            ...state.report,
            elements: [...state.report.elements, newElement]
          },
          isDirty: true
        }));
      },

      updateElement: (elementId: string, updates: Partial<ReportElement>) => {
        set(state => {
          const updatedElements = state.report.elements.map(element =>
            element.id === elementId ? { ...element, ...updates } : element
          );

          return {
            report: {
              ...state.report,
              elements: updatedElements
            },
            isDirty: true
          };
        });
      },

      removeElement: (elementId: string) => {
        set(state => ({
          report: {
            ...state.report,
            elements: state.report.elements.filter(element => element.id !== elementId)
          },
          isDirty: true
        }));
      },

      selectCanvasElement: (elementId?: string) => {
        // This method doesn't modify state, but could be used to track selected element
        // if needed in the future
      },

      clearError: () => set({ error: null })
    }),
    { name: 'report-builder-store' }
  )
);
