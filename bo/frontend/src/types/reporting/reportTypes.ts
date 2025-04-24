// Report Types
export interface Report {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  lastRunAt?: string;
  createdBy: string;
  dataSourceId: string;
  data_source_id?: string; // Used in the API
  isPublic: boolean;
  isFavorite: boolean;
  elements: ReportElement[];
  globalFilters?: FilterCondition[];
  parameters?: Record<string, any>;
  // Additional properties used in the report builder
  configuration?: {
    reportType?: string;
    fields?: ReportField[];
    filters?: ReportFilter[];
    sorting?: ReportSorting[];
    visualizations?: VisualizationConfig[];
    layout?: {
      items: LayoutItem[];
    };
  };
}

// Data Source Types
export interface DataSource {
  id: string;
  name: string;
  description?: string;
  connectionType: string;
  connectionConfig: Record<string, any>;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  // Additional properties used in the UI
  type?: string;
  host?: string;
  port?: string;
  database?: string;
  username?: string;
  password?: string;
  connection_string?: string;
  status?: string;
}

// Report Element Types
export interface ReportElement {
  id: string;
  name: string;
  reportId: string;
  dimensions: string[];
  metrics: string[];
  filters?: FilterCondition[];
  sort?: SortCondition[];
  visualization: Visualization;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// Visualization Types
export interface Visualization {
  type: VisualizationType;
  title?: string;
  subtitle?: string;
  config?: Record<string, any>;
}

export enum VisualizationType {
  TABLE = 'table',
  BAR_CHART = 'bar_chart',
  LINE_CHART = 'line_chart',
  PIE_CHART = 'pie_chart',
  AREA_CHART = 'area_chart',
  SCATTER_CHART = 'scatter_chart',
  CARD = 'card',
}

// Filter Types
export interface FilterCondition {
  id: string;
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  operator: FilterOperator;
  value: any;
}

export enum FilterOperator {
  EQUALS = 'equals',
  NOT_EQUALS = 'not_equals',
  GREATER_THAN = 'greater_than',
  LESS_THAN = 'less_than',
  GREATER_THAN_OR_EQUALS = 'greater_than_or_equals',
  LESS_THAN_OR_EQUALS = 'less_than_or_equals',
  CONTAINS = 'contains',
  NOT_CONTAINS = 'not_contains',
  STARTS_WITH = 'starts_with',
  ENDS_WITH = 'ends_with',
  IN = 'in',
  NOT_IN = 'not_in',
  BETWEEN = 'between',
  IS_NULL = 'is_null',
  IS_NOT_NULL = 'is_not_null',
}

// Sort Types
export interface SortCondition {
  id: string;
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  direction: 'asc' | 'desc';
}

// Field Types
export interface Field {
  id: string;
  name: string;
  displayName: string;
  dataType: string;
  description?: string;
}

export interface Dimension extends Field {
  isTimeDimension?: boolean;
  timeGranularity?: 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour' | 'minute' | 'second';
}

export interface Metric extends Field {
  aggregation: 'sum' | 'avg' | 'min' | 'max' | 'count' | 'count_distinct';
}

export interface CalculatedMetric extends Field {
  formula: string;
  referencedFields: string[];
}

// Report Data Types
export interface ReportData {
  reportId: string;
  generatedAt: string;
  data: any[];
  metadata?: Record<string, any>;
}

// Chart Data Types
export interface ChartData {
  [key: string]: any;
}

export interface VisualizationConfig {
  id?: string;
  type?: string;
  settings?: Record<string, any>;
  title?: string;
  subtitle?: string;
  measures?: Array<{
    field: { id: string; name: string };
    label?: string;
    color?: string;
  }>;
  dimension?: {
    field: { id: string; name: string };
  };
  dimensions?: string[];
  metrics?: string[];
  chartSubtype?: 'stacked' | 'curved' | 'dotted' | 'compact' | 'donut';
  backgroundColor?: string;
  textColor?: string;
  format?: {
    decimals?: number;
    prefix?: string;
    suffix?: string;
    thousandsSeparator?: string;
  };
  trend?: {
    compare: 'previous' | 'target';
    previousValue?: number;
    targetValue?: number;
  };
  colors?: string[];
  // Table chart specific properties
  pageSize?: number;
  showSearch?: boolean;
  showPagination?: boolean;
  showDownload?: boolean;
  sortable?: boolean;
  formatting?: Record<string, any>;
  columnWidths?: Record<string, number>;
  columnOrder?: string[];
}

// Report Creation/Update Types
export interface ReportCreate {
  name: string;
  description?: string;
  dataSourceId: string;
  isPublic?: boolean;
  elements: ReportElementCreate[];
  globalFilters?: FilterConditionCreate[];
  parameters?: Record<string, any>;
}

export interface ReportUpdate {
  name?: string;
  description?: string;
  isPublic?: boolean;
  elements?: ReportElementCreate[];
  globalFilters?: FilterConditionCreate[];
  parameters?: Record<string, any>;
}

export interface ReportElementCreate {
  name: string;
  dimensions: string[];
  metrics: string[];
  filters?: FilterConditionCreate[];
  sort?: SortConditionCreate[];
  visualization: VisualizationCreate;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface VisualizationCreate {
  type: VisualizationType;
  title?: string;
  subtitle?: string;
  config?: Record<string, any>;
}

export interface FilterConditionCreate {
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  operator: FilterOperator;
  value: any;
}

export interface SortConditionCreate {
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  direction: 'asc' | 'desc';
}

// Report Builder Types
export interface ReportField {
  id: string;
  name: string;
  type: string;
  dataType?: string;
  aggregation?: string;
  formula?: string;
  displayName?: string;
  description?: string;
  format?: {
    type?: string;
    precision?: number;
    prefix?: string;
    suffix?: string;
  };
}

export interface ReportFilter {
  id: string;
  field: {
    id: string;
    name: string;
    type: string;
  };
  operator: string;
  value: any;
  displayValue?: string;
  isActive?: boolean;
}

export interface ReportSorting {
  id: string;
  field: {
    id: string;
    name: string;
  };
  direction: 'asc' | 'desc';
}

export interface LayoutItem {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  content: {
    type: string;
    visualizationId?: string;
    textContent?: string;
  };
}

export interface ReportExecutionResult {
  id: string;
  reportId: string;
  executedAt: string;
  data: any[];
  metadata?: Record<string, any>;
}

// Report Response Types
export interface ReportResponse {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  lastRunAt?: string;
  createdBy: string;
  dataSourceId: string;
  isPublic: boolean;
  isFavorite: boolean;
  elements: ReportElementResponse[];
  globalFilters?: FilterConditionResponse[];
  parameters?: Record<string, any>;
}

export interface ReportElementResponse {
  id: string;
  name: string;
  reportId: string;
  dimensions: string[];
  metrics: string[];
  filters?: FilterConditionResponse[];
  sort?: SortConditionResponse[];
  visualization: VisualizationResponse;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface VisualizationResponse {
  type: VisualizationType;
  title?: string;
  subtitle?: string;
  config?: Record<string, any>;
}

export interface FilterConditionResponse {
  id: string;
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  operator: FilterOperator;
  value: any;
}

export interface SortConditionResponse {
  id: string;
  fieldId: string;
  fieldType: 'dimension' | 'metric';
  direction: 'asc' | 'desc';
}
