import axios from '../utils/axios'; // Using configured axios instance with auth headers
import {
  DataSource,
  Dimension,
  Metric,
  CalculatedMetric,
  Report,
  ReportPermission,
  ReportExecutionResult
} from '../types/reporting';

const API_BASE = '/reporting';  // Remove the '/api' prefix to match the backend path

// Data Sources
export const getDataSources = async (): Promise<DataSource[]> => {
  const response = await axios.get(`${API_BASE}/datasources/`);
  return response.data;
};

export const getDataSource = async (id: string): Promise<DataSource> => {
  const response = await axios.get(`${API_BASE}/datasources/${id}`);
  return response.data;
};

export const createDataSource = async (dataSource: Omit<DataSource, 'id' | 'creator_id' | 'created_at' | 'updated_at' | 'is_active'>): Promise<DataSource> => {
  const response = await axios.post(`${API_BASE}/datasources/`, dataSource);
  return response.data;
};

export const updateDataSource = async (id: string, dataSource: Partial<DataSource>): Promise<DataSource> => {
  const response = await axios.put(`${API_BASE}/datasources/${id}`, dataSource);
  return response.data;
};

export const deleteDataSource = async (id: string): Promise<void> => {
  await axios.delete(`${API_BASE}/datasources/${id}`);
};

// Dimensions
export const getDimensions = async (dataSourceId?: string): Promise<Dimension[]> => {
  const params = dataSourceId ? { data_source_id: dataSourceId } : {};
  const response = await axios.get(`${API_BASE}/dimensions/`, { params });
  return response.data;
};

export const createDimension = async (dimension: Omit<Dimension, 'id' | 'created_at' | 'updated_at' | 'is_active'>): Promise<Dimension> => {
  const response = await axios.post(`${API_BASE}/dimensions/`, dimension);
  return response.data;
};

// Metrics
export const getMetrics = async (dataSourceId?: string): Promise<Metric[]> => {
  const params = dataSourceId ? { data_source_id: dataSourceId } : {};
  const response = await axios.get(`${API_BASE}/metrics/`, { params });
  return response.data;
};

export const createMetric = async (metric: Omit<Metric, 'id' | 'created_at' | 'updated_at' | 'is_active'>): Promise<Metric> => {
  const response = await axios.post(`${API_BASE}/metrics/`, metric);
  return response.data;
};

// Calculated Metrics
export const getCalculatedMetrics = async (dataSourceId?: string): Promise<CalculatedMetric[]> => {
  const params = dataSourceId ? { data_source_id: dataSourceId } : {};
  const response = await axios.get(`${API_BASE}/calculated_metrics/`, { params });
  return response.data;
};

export const createCalculatedMetric = async (metric: Omit<CalculatedMetric, 'id' | 'created_at' | 'updated_at' | 'is_active'>): Promise<CalculatedMetric> => {
  const response = await axios.post(`${API_BASE}/calculated_metrics/`, metric);
  return response.data;
};

// Reports
export const getReports = async (params?: { data_source_id?: string; is_public?: boolean; is_favorite?: boolean }): Promise<Report[]> => {
  const response = await axios.get(`${API_BASE}/reports/`, { params });
  return response.data;
};

export const getReport = async (id: string): Promise<Report> => {
  const response = await axios.get(`${API_BASE}/reports/${id}`);
  return response.data;
};

export const createReport = async (report: Omit<Report, 'id' | 'creator_id' | 'created_at' | 'updated_at' | 'last_run_at'>): Promise<Report> => {
  const response = await axios.post(`${API_BASE}/reports/`, report);
  return response.data;
};

export const updateReport = async (id: string, report: Partial<Report>): Promise<Report> => {
  const response = await axios.put(`${API_BASE}/reports/${id}`, report);
  return response.data;
};

export const deleteReport = async (id: string): Promise<void> => {
  await axios.delete(`${API_BASE}/reports/${id}`);
};

// Report permissions
export const getReportPermissions = async (reportId: string): Promise<ReportPermission[]> => {
  const response = await axios.get(`${API_BASE}/reports/${reportId}/permissions`);
  return response.data;
};

export const addReportPermission = async (reportId: string, permission: Omit<ReportPermission, 'id' | 'report_id' | 'created_at'>): Promise<ReportPermission> => {
  const response = await axios.post(`${API_BASE}/reports/${reportId}/permissions`, permission);
  return response.data;
};

export const removeReportPermission = async (reportId: string, permissionId: number): Promise<void> => {
  await axios.delete(`${API_BASE}/reports/${reportId}/permissions/${permissionId}`);
};

// Report execution
export const runReport = async (reportId: string, params?: Record<string, any>): Promise<ReportExecutionResult> => {
  const response = await axios.post(`${API_BASE}/reports/${reportId}/run`, params || {});
  return response.data;
};