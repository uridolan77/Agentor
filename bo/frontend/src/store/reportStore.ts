import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  DataSource,
  Dimension,
  Metric,
  CalculatedMetric,
  Report,
  ReportField,
  ReportFilter,
  ReportSorting,
  VisualizationConfig,
  LayoutItem,
  ReportExecutionResult
} from '../types/reporting'; // Corrected import path
import * as reportApi from '../api/reportApi';
import {
  adaptApiResponse,
  adaptApiResponseArray,
  snakeToCamelReport,
  camelToSnakeReport,
  snakeToCamelDataSource,
  snakeToCamelReportExecutionResult
} from '../utils/typeAdapters';

interface ReportState {
  // Data sources
  dataSources: DataSource[];
  selectedDataSourceId: string | null;
  loadingDataSources: boolean;

  // Fields (dimensions, metrics)
  dimensions: Dimension[];
  metrics: Metric[];
  calculatedMetrics: CalculatedMetric[];
  loadingFields: boolean;

  // Reports
  reports: Report[];
  selectedReport: Report | null;
  loadingReports: boolean;

  // Report builder state
  reportBuilderFields: ReportField[];
  reportBuilderFilters: ReportFilter[];
  reportBuilderSorting: ReportSorting[];
  reportBuilderVisualizations: VisualizationConfig[];
  reportBuilderLayout: LayoutItem[];
  reportName: string;
  reportDescription: string;

  // Report execution
  reportExecutionResult: ReportExecutionResult | null;
  executingReport: boolean;

  // Actions
  fetchDataSources: () => Promise<void>;
  selectDataSource: (dataSourceId: string | null) => void;
  fetchFields: (dataSourceId: string) => Promise<void>;

  fetchReports: (filters?: { dataSourceId?: string, isPublic?: boolean, isFavorite?: boolean }) => Promise<void>;
  fetchReportById: (reportId: string) => Promise<void>;
  createReport: () => Promise<Report>;
  updateReport: () => Promise<Report>;
  deleteReport: (reportId: string) => Promise<void>;

  // Report builder actions
  setReportName: (name: string) => void;
  setReportDescription: (description: string) => void;
  addField: (field: ReportField) => void;
  removeField: (fieldId: string) => void;
  addFilter: (filter: ReportFilter) => void;
  updateFilter: (index: number, filter: ReportFilter) => void;
  removeFilter: (index: number) => void;
  addSorting: (sorting: ReportSorting) => void;
  updateSorting: (index: number, sorting: ReportSorting) => void;
  removeSorting: (index: number) => void;
  addVisualization: (visualization: VisualizationConfig) => void;
  updateVisualization: (id: string, visualization: Partial<VisualizationConfig>) => void;
  removeVisualization: (id: string) => void;
  updateLayout: (layout: LayoutItem[]) => void;

  // Report execution actions
  executeReport: (reportId?: string, params?: Record<string, any>) => Promise<void>;
  resetReportBuilder: () => void;
}

// Use type assertion to fix the persist middleware error
const useReportStore = create<ReportState>(
  (persist(
    (set, get) => ({
      // Initial state
      dataSources: [],
      selectedDataSourceId: null,
      loadingDataSources: false,

      dimensions: [],
      metrics: [],
      calculatedMetrics: [],
      loadingFields: false,

      reports: [],
      selectedReport: null,
      loadingReports: false,

      reportBuilderFields: [],
      reportBuilderFilters: [],
      reportBuilderSorting: [],
      reportBuilderVisualizations: [],
      reportBuilderLayout: [],
      reportName: 'New Report',
      reportDescription: '',

      reportExecutionResult: null,
      executingReport: false,

      // Actions
      fetchDataSources: async () => {
        set({ loadingDataSources: true });
        try {
          const apiDataSources = await reportApi.getDataSources();
          // Convert API response to the expected format
          const dataSources = adaptApiResponseArray(apiDataSources, snakeToCamelDataSource);
          set({ dataSources, loadingDataSources: false });
        } catch (error) {
          console.error('Error fetching data sources:', error);
          set({ loadingDataSources: false });
        }
      },

      selectDataSource: (dataSourceId: string | null) => {
        set({ selectedDataSourceId: dataSourceId });
        if (dataSourceId) {
          (get() as ReportState).fetchFields(dataSourceId);
        } else {
          set({ dimensions: [], metrics: [], calculatedMetrics: [] });
        }
      },

      fetchFields: async (dataSourceId: string) => {
        set({ loadingFields: true });
        try {
          // Fetch dimensions, metrics, and calculated metrics in parallel
          const [dimensions, metrics, calculatedMetrics] = await Promise.all([
            reportApi.getDimensions(dataSourceId),
            reportApi.getMetrics(dataSourceId),
            reportApi.getCalculatedMetrics(dataSourceId)
          ]);

          set({ dimensions, metrics, calculatedMetrics, loadingFields: false });
        } catch (error) {
          console.error('Error fetching fields:', error);
          set({ loadingFields: false });
        }
      },

      fetchReports: async (filters: { dataSourceId?: string, isPublic?: boolean, isFavorite?: boolean } = {}) => {
        set({ loadingReports: true });
        try {
          const apiReports = await reportApi.getReports({
            data_source_id: filters.dataSourceId,
            is_public: filters.isPublic,
            is_favorite: filters.isFavorite
          });
          // Convert API response to the expected format
          const reports = adaptApiResponseArray(apiReports, snakeToCamelReport);
          set({ reports, loadingReports: false });
        } catch (error) {
          console.error('Error fetching reports:', error);
          set({ loadingReports: false });
        }
      },

      fetchReportById: async (reportId: string) => {
        try {
          const apiReport = await reportApi.getReport(reportId);
          // Convert API response to the expected format
          const report = snakeToCamelReport(apiReport);
          set({ selectedReport: report });

          // Set report builder state from the selected report
          set({
            reportName: report.name,
            reportDescription: report.description || '',
            reportBuilderFields: report.configuration?.fields || [],
            reportBuilderFilters: report.configuration?.filters || [],
            reportBuilderSorting: report.configuration?.sorting || [],
            reportBuilderVisualizations: report.configuration?.visualizations || [],
            reportBuilderLayout: report.configuration?.layout?.items || [],
            selectedDataSourceId: report.dataSourceId,
          });

          // Fetch fields for the report's data source
          if (report.dataSourceId) {
            (get() as ReportState).fetchFields(report.dataSourceId);
          }

        } catch (error) {
          console.error('Error fetching report:', error);
        }
      },

      createReport: async () => {
        const state = get() as ReportState;
        const {
          reportName,
          reportDescription,
          selectedDataSourceId,
          reportBuilderFields,
          reportBuilderFilters,
          reportBuilderSorting,
          reportBuilderVisualizations,
          reportBuilderLayout
        } = state;

        if (!selectedDataSourceId) {
          throw new Error('No data source selected');
        }

        // Create a report object in the camelCase format
        // Use type assertion to tell TypeScript this is a partial Report object
        const reportData = {
          name: reportName,
          description: reportDescription,
          dataSourceId: selectedDataSourceId,
          configuration: {
            reportType: 'custom',
            fields: reportBuilderFields,
            filters: reportBuilderFilters,
            sorting: reportBuilderSorting,
            visualizations: reportBuilderVisualizations,
            layout: {
              items: reportBuilderLayout
            }
          },
          isPublic: false,
          isFavorite: false,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          createdBy: '',
          elements: []
        };

        try {
          // Convert to API format before sending, specifying this is not an update
          // Use type assertion to avoid TypeScript errors
          const apiReportData = camelToSnakeReport(reportData as any, false);
          const createdApiReport = await reportApi.createReport(apiReportData);
          // Convert API response back to the expected format
          const createdReport = snakeToCamelReport(createdApiReport);
          
          // Update reports list
          set((state: ReportState) => ({
            reports: [...state.reports, createdReport],
            selectedReport: createdReport
          }));
          return createdReport;
        } catch (error) {
          console.error('Error creating report:', error);
          throw error;
        }
      },

      updateReport: async () => {
        const state = get() as ReportState;
        const {
          selectedReport,
          reportName,
          reportDescription,
          reportBuilderFields,
          reportBuilderFilters,
          reportBuilderSorting,
          reportBuilderVisualizations,
          reportBuilderLayout
        } = state;

        if (!selectedReport) {
          throw new Error('No report selected for update');
        }

        // Create a report update object in the camelCase format
        const reportUpdateData = {
          ...selectedReport,
          name: reportName,
          description: reportDescription,
          configuration: {
            reportType: 'custom',
            fields: reportBuilderFields,
            filters: reportBuilderFilters,
            sorting: reportBuilderSorting,
            visualizations: reportBuilderVisualizations,
            layout: {
              items: reportBuilderLayout
            }
          },
          updatedAt: new Date().toISOString()
        };

        try {
          // Convert to API format before sending, specifying this is an update
          // Use type assertion to avoid TypeScript errors
          const apiReportUpdateData = camelToSnakeReport(reportUpdateData as any, true);
          const updatedApiReport = await reportApi.updateReport(selectedReport.id, apiReportUpdateData);
          // Convert API response back to the expected format
          const updatedReport = snakeToCamelReport(updatedApiReport);
          
          // Update reports list
          set((state: ReportState) => ({
            reports: state.reports.map((r: Report) => r.id === updatedReport.id ? updatedReport : r),
            selectedReport: updatedReport
          }));
          return updatedReport;
        } catch (error) {
          console.error('Error updating report:', error);
          throw error;
        }
      },

      deleteReport: async (reportId: string) => {
        try {
          await reportApi.deleteReport(reportId);
          // Update reports list
          set((state: ReportState) => ({
            reports: state.reports.filter((r: Report) => r.id !== reportId),
            selectedReport: state.selectedReport?.id === reportId ? null : state.selectedReport
          }));
        } catch (error) {
          console.error('Error deleting report:', error);
          throw error;
        }
      },

      // Report builder actions
      setReportName: (name: string) => set({ reportName: name }),
      setReportDescription: (description: string) => set({ reportDescription: description }),

      addField: (field: ReportField) => set((state: ReportState) => ({
        reportBuilderFields: [...state.reportBuilderFields, field]
      })),

      removeField: (fieldId: string) => set((state: ReportState) => ({
        reportBuilderFields: state.reportBuilderFields.filter(f => f.id !== fieldId)
      })),

      addFilter: (filter: ReportFilter) => set((state: ReportState) => ({
        reportBuilderFilters: [...state.reportBuilderFilters, filter]
      })),

      updateFilter: (index: number, filter: ReportFilter) => set((state: ReportState) => {
        const updatedFilters = [...state.reportBuilderFilters];
        updatedFilters[index] = filter;
        return { reportBuilderFilters: updatedFilters };
      }),

      removeFilter: (index: number) => set((state: ReportState) => {
        const updatedFilters = [...state.reportBuilderFilters];
        updatedFilters.splice(index, 1);
        return { reportBuilderFilters: updatedFilters };
      }),

      addSorting: (sorting: ReportSorting) => set((state: ReportState) => ({
        reportBuilderSorting: [...state.reportBuilderSorting, sorting]
      })),

      updateSorting: (index: number, sorting: ReportSorting) => set((state: ReportState) => {
        const updatedSorting = [...state.reportBuilderSorting];
        updatedSorting[index] = sorting;
        return { reportBuilderSorting: updatedSorting };
      }),

      removeSorting: (index: number) => set((state: ReportState) => {
        const updatedSorting = [...state.reportBuilderSorting];
        updatedSorting.splice(index, 1);
        return { reportBuilderSorting: updatedSorting };
      }),

      addVisualization: (visualization: VisualizationConfig) => set((state: ReportState) => ({
        reportBuilderVisualizations: [...state.reportBuilderVisualizations, visualization]
      })),

      updateVisualization: (id: string, visualizationUpdate: Partial<VisualizationConfig>) => set((state: ReportState) => ({
        reportBuilderVisualizations: state.reportBuilderVisualizations.map(v =>
          v.id === id ? { ...v, ...visualizationUpdate } : v
        )
      })),

      removeVisualization: (id: string) => set((state: ReportState) => ({
        reportBuilderVisualizations: state.reportBuilderVisualizations.filter(v => v.id !== id),
        reportBuilderLayout: state.reportBuilderLayout.filter(item =>
          item.content.type !== 'visualization' || item.content.visualizationId !== id
        )
      })),

      updateLayout: (layout: LayoutItem[]) => set({ reportBuilderLayout: layout }),

      executeReport: async (reportId?: string, params?: Record<string, any>) => {
        const state = get() as ReportState;
        const id = reportId || state.selectedReport?.id;
        if (!id) {
          throw new Error('No report selected for execution');
        }

        set({ executingReport: true });
        try {
          const apiResult = await reportApi.runReport(id, params);
          // Convert API response to the expected format
          const result = snakeToCamelReportExecutionResult(apiResult);
          set({ reportExecutionResult: result, executingReport: false });
        } catch (error) {
          console.error('Error executing report:', error);
          set({ executingReport: false });
          throw error;
        }
      },

      resetReportBuilder: () => {
        set({
          selectedReport: null,
          reportName: 'New Report',
          reportDescription: '',
          reportBuilderFields: [],
          reportBuilderFilters: [],
          reportBuilderSorting: [],
          reportBuilderVisualizations: [],
          reportBuilderLayout: [],
          reportExecutionResult: null
        });
      }
    }),
    {
      name: 'report-store',
      // Only persist selected items, not the entire state
      partialize: (state: ReportState) => ({
        selectedDataSourceId: state.selectedDataSourceId,
        reportName: state.reportName,
        reportDescription: state.reportDescription,
        reportBuilderFields: state.reportBuilderFields,
        reportBuilderFilters: state.reportBuilderFilters,
        reportBuilderSorting: state.reportBuilderSorting,
        reportBuilderVisualizations: state.reportBuilderVisualizations,
        reportBuilderLayout: state.reportBuilderLayout
      })
    }
  ) as any)
);

export default useReportStore;
