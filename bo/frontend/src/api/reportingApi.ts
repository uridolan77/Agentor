import axios from '../utils/axios'; // Using configured axios instance with auth headers
import {
  Report,
  DataSource,
  Dimension,
  Metric,
  CalculatedMetric,
  ReportData,
  ReportCreate,
  ReportUpdate
} from '../types/reporting/reportTypes';

// API endpoints
const endpoints = {
  // Reports
  reports: '/reporting/reports',
  report: (id: string) => `/reporting/reports/${id}`,
  reportData: (id: string) => `/reporting/reports/${id}/run`,
  reportFavorite: (id: string) => `/reporting/reports/${id}`,
  reportClone: (id: string) => `/reporting/reports/${id}/clone`,

  // Data Sources
  dataSources: '/reporting/datasources',
  dataSource: (id: string) => `/reporting/datasources/${id}`,
  dataSourceFields: (id: string) => `/reporting/datasources/${id}/fields`,

  // Fields
  dimensions: '/reporting/dimensions',
  metrics: '/reporting/metrics',
  calculatedMetrics: '/reporting/calculated-metrics',
};

// API methods
const reportingAPI = {
  // Reports
  getReports: async (filters?: Record<string, any>): Promise<Report[]> => {
    const { data } = await axios.get(endpoints.reports, { params: filters });
    return data;
  },

  getReport: async (id: string): Promise<Report> => {
    const { data } = await axios.get(endpoints.report(id));
    return data;
  },

  createReport: async (report: ReportCreate): Promise<Report> => {
    const { data } = await axios.post(endpoints.reports, report);
    return data;
  },

  updateReport: async (id: string, report: ReportUpdate): Promise<Report> => {
    const { data } = await axios.put(endpoints.report(id), report);
    return data;
  },

  deleteReport: async (id: string): Promise<void> => {
    await axios.delete(endpoints.report(id));
  },

  generateReportData: async (params: { reportId: string; parameters?: Record<string, any> }): Promise<ReportData> => {
    const { data } = await axios.post(endpoints.reportData(params.reportId), params.parameters || {});
    return data;
  },

  toggleFavoriteReport: async (id: string, isFavorite: boolean): Promise<Report> => {
    const { data } = await axios.put(endpoints.reportFavorite(id), { is_favorite: isFavorite });
    return data;
  },

  cloneReport: async (id: string): Promise<Report> => {
    const { data } = await axios.post(endpoints.reportClone(id));
    return data;
  },

  // Data Sources
  getDataSources: async (): Promise<DataSource[]> => {
    const { data } = await axios.get(endpoints.dataSources);
    return data;
  },

  getDataSource: async (id: string): Promise<DataSource> => {
    const { data } = await axios.get(endpoints.dataSource(id));
    return data;
  },

  createDataSource: async (dataSource: any): Promise<DataSource> => {
    const { data } = await axios.post(endpoints.dataSources, dataSource);
    return data;
  },

  updateDataSource: async (id: string, dataSource: any): Promise<DataSource> => {
    const { data } = await axios.put(endpoints.dataSource(id), dataSource);
    return data;
  },

  deleteDataSource: async (id: string): Promise<void> => {
    await axios.delete(endpoints.dataSource(id));
  },

  getDataSourceFields: async (id: string): Promise<{
    dimensions: Dimension[];
    metrics: Metric[];
    calculatedMetrics: CalculatedMetric[];
  }> => {
    const { data } = await axios.get(endpoints.dataSourceFields(id));
    return data;
  },

  // Fields
  getDimensions: async (): Promise<Dimension[]> => {
    const { data } = await axios.get(endpoints.dimensions);
    return data;
  },

  getMetrics: async (): Promise<Metric[]> => {
    const { data } = await axios.get(endpoints.metrics);
    return data;
  },

  getCalculatedMetrics: async (): Promise<CalculatedMetric[]> => {
    const { data } = await axios.get(endpoints.calculatedMetrics);
    return data;
  },
};

export default reportingAPI;
