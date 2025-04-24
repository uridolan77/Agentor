export interface DataSource {
  id: string;
  name: string;
  description?: string;
  connection_type: string; // This maps to 'type' in the form
  connection_config: Record<string, any>;
  creator_id: number;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  status?: string;
  // Additional form fields that will be mapped to connection_config
  type?: string;
  host?: string;
  port?: string;
  database?: string;
  username?: string;
  password?: string;
  connection_string?: string;
  ssl_enabled?: boolean;
  options?: Record<string, any>;
}

export enum DataSourceType {
  MySQL = 'mysql',
  PostgreSQL = 'postgresql',
  SQLite = 'sqlite',
  CSV = 'csv',
  Excel = 'excel',
  ConnectionString = 'connection_string'
}

export interface Dimension {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  data_source_id: string;
  data_type: string;
  table_name: string;
  column_name: string;
  formatting?: Record<string, any>;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface Metric {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  data_source_id: string;
  data_type: string;
  calculation_type: string;
  table_name: string;
  column_name: string;
  formatting?: Record<string, any>;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface CalculatedMetric {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  data_source_id: string;
  data_type: string;
  formula: string;
  dependencies: string[];
  formatting?: Record<string, any>;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface Report {
  id: string;
  name: string;
  description?: string;
  dataSourceId: string;
  configuration: ReportConfiguration;
  creatorId: number;
  createdAt: string;
  updatedAt: string;
  isPublic: boolean;
  isFavorite: boolean;
  lastRunAt?: string;
  thumbnail?: string;
}

export interface ReportPermission {
  id: number;
  report_id: string;
  user_id?: number;
  team_id?: number;
  permission_type: string;
  created_at: string;
}

export interface ReportConfiguration {
  reportType: string;
  fields: ReportField[];
  filters?: ReportFilter[];
  sorting?: ReportSorting[];
  limit?: number;
  offset?: number;
  visualizations: VisualizationConfig[];
  layout?: LayoutConfig;
}

export interface ReportField {
  id: string;
  type: 'dimension' | 'metric' | 'calculated_metric';
  name: string;
}

export interface ReportFilter {
  column: string;
  operator: string;
  value: any;
}

export interface ReportSorting {
  column: string;
  direction: 'asc' | 'desc';
}

export interface VisualizationConfig {
  id: string;
  type: 'table' | 'bar' | 'line' | 'pie' | 'area' | 'scatter' | 'metric';
  title: string;
  dimensions: string[];
  metrics: string[];
  settings: Record<string, any>;
}

export interface LayoutConfig {
  items: LayoutItem[];
}

export interface LayoutItem {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  content: {
    type: 'visualization';
    visualizationId: string;
  };
}

export interface ReportExecutionResult {
  id: string;
  report_id: string;
  timestamp: string;
  status: 'success' | 'error';
  data: {
    columns: string[];
    rows: Record<string, any>[];
  };
  params: Record<string, any>;
  error?: string;
}

// Data Canvas Types
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

export interface TableSchema {
  name: string;
  columns: ColumnSchema[];
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
