import {
  DataSource as CanonicalDataSource, // Renamed alias
  Report as CanonicalReport, // Renamed alias
  ReportConfiguration as CanonicalReportConfiguration, // Renamed alias
  ReportField as CanonicalReportField, // Renamed alias
  ReportFilter as CanonicalReportFilter, // Renamed alias
  ReportSorting as CanonicalReportSorting, // Renamed alias
  VisualizationConfig as CanonicalVisualizationConfig, // Renamed alias
  LayoutItem as CanonicalLayoutItem, // Renamed alias
  ReportExecutionResult as CanonicalReportExecutionResult // Renamed alias
} from '../types/reporting';

// Keep old types for reference if needed by adapters initially
import {
  DataSource as OldDataSourceType, // Renamed alias
  Report as OldReportType, // Renamed alias
  ReportField as OldReportFieldType, // Renamed alias
  ReportFilter as OldReportFilterType, // Renamed alias
  ReportSorting as OldReportSortingType, // Renamed alias
  VisualizationConfig as OldVisualizationConfigType, // Renamed alias
  LayoutItem as OldLayoutItemType, // Renamed alias
  ReportExecutionResult as OldReportExecutionResultType // Renamed alias
} from '../types/reporting/reportTypes'; // Path to the old types

/**
 * Type adapter functions to convert between snake_case and camelCase data structures
 * Type adapter functions to convert between snake_case API structures and canonical camelCase types.
 */

// DataSource adapters
// Takes snake_case API response, returns CanonicalDataSource (camelCase)
export function snakeToCamelDataSource(source: any): CanonicalDataSource {
  // Assuming API returns snake_case keys corresponding to CanonicalDataSource
  return {
    id: source.id, // Assuming id is same
    name: source.name, // Assuming name is same
    description: source.description, // Assuming description is same
    connection_type: source.connection_type, // Map snake_case from API to type
    connection_config: source.connection_config, // Map snake_case from API to type
    creator_id: source.creator_id, // Map snake_case from API to type
    created_at: source.created_at, // Map snake_case from API to type
    updated_at: source.updated_at, // Map snake_case from API to type
    is_active: source.is_active, // Map snake_case from API to type
    status: source.status || 'unknown', // Default status if not provided
    // Map additional form fields if they exist in the API response
    type: source.type,
    host: source.host,
    port: source.port,
    database: source.database,
    username: source.username,
    password: source.password, // Be careful with passwords
    connection_string: source.connection_string,
    ssl_enabled: source.ssl_enabled,
    options: source.options,
  };
}

// Takes CanonicalDataSource (camelCase), returns snake_case object for API
export function camelToSnakeDataSource(source: CanonicalDataSource): any {
  // Map CanonicalDataSource fields to snake_case for the API
  return {
    id: source.id, // Pass id if needed
    name: source.name,
    description: source.description,
    connection_type: source.connection_type, // Map camelCase to snake_case
    connection_config: source.connection_config, // Map camelCase to snake_case
    creator_id: source.creator_id, // Map camelCase to snake_case
    // created_at and updated_at are usually set by backend, no need to send
    is_active: source.is_active, // Map camelCase to snake_case
    status: source.status,
    // Map additional form fields back to snake_case if needed by API
    type: source.type,
    host: source.host,
    port: source.port,
    database: source.database,
    username: source.username,
    password: source.password,
    connection_string: source.connection_string,
    ssl_enabled: source.ssl_enabled,
    options: source.options,
  };
}

// Report adapters
// Takes snake_case API response, returns CanonicalReport (camelCase)
export function snakeToCamelReport(report: any): CanonicalReport { // Output is CanonicalReport
  return {
    id: report.id,
    name: report.name,
    description: report.description,
    dataSourceId: report.data_source_id, // snake_case from API
    creatorId: report.creator_id, // snake_case from API
    createdAt: report.created_at, // snake_case from API
    updatedAt: report.updated_at, // snake_case from API
    isPublic: report.is_public, // snake_case from API
    isFavorite: report.is_favorite, // snake_case from API
    lastRunAt: report.last_run_at, // snake_case from API
    thumbnail: report.thumbnail, // Assuming API might send this
    // Map configuration (assuming configuration from API is already camelCase or needs its own adapter)
    // For simplicity, let's assume configuration is passed through if it exists,
    // or provide a default structure matching the Report type.
    configuration: report.configuration ? {
      reportType: report.configuration.reportType, // Assuming camelCase in API config
      // Assuming fields, filters etc in API config are snake_case and need adapting
      // OR assuming they are already camelCase matching the Report type
      // Let's assume they match the Report type for now
      fields: report.configuration.fields || [],
      filters: report.configuration.filters || [],
      sorting: report.configuration.sorting || [],
      visualizations: report.configuration.visualizations || [],
      layout: report.configuration.layout || { items: [] } // Ensure layout exists
    } : { // Default configuration if not provided by API
      reportType: 'custom',
      fields: [],
      filters: [],
      sorting: [],
      visualizations: [],
      layout: { items: [] }
    }
  };
}

// Takes CanonicalReport (camelCase), returns snake_case object for API
export function camelToSnakeReport(report: CanonicalReport, isUpdate: boolean = false): any { // Input is CanonicalReport
  const snakeReport: any = {
    name: report.name,
    description: report.description,
    data_source_id: report.dataSourceId, // camelCase to snake_case
    // creator_id might be set by the backend based on auth context
    created_at: report.createdAt, // Usually set by backend
    updated_at: report.updatedAt, // Usually set by backend
    is_public: report.isPublic, // camelCase to snake_case
    is_favorite: report.isFavorite, // camelCase to snake_case
    last_run_at: report.lastRunAt, // camelCase to snake_case
    thumbnail: report.thumbnail,
    // Map configuration (assuming API expects camelCase configuration matching the Report type for now)
    configuration: report.configuration || {
      reportType: 'custom',
      fields: [],
      filters: [],
      sorting: [],
      visualizations: [],
      layout: { items: [] }
    }
  };

  // Include id and potentially creator_id only for updates
  if (isUpdate) {
    snakeReport.id = report.id;
    // If creatorId needs to be sent for updates, include it here
    // snakeReport.creator_id = report.creatorId;
  }
  // We might need to explicitly send creator_id for creation too, depending on API
  // If so, uncomment below and ensure report.creatorId has a value
  // else {
  //   snakeReport.creator_id = report.creatorId;
  // }


  return snakeReport;
}

// ReportField adapters
// Takes snake_case API field, returns CanonicalReportField
export function snakeToCamelReportField(field: any): CanonicalReportField {
  return {
    id: field.id,
    name: field.name,
    type: field.type
  };
}

// Takes CanonicalReportField, returns snake_case object for API
export function camelToSnakeReportField(field: CanonicalReportField): any {
  return {
    id: field.id,
    name: field.name,
    type: field.type as 'dimension' | 'metric' | 'calculated_metric'
  };
}

// ReportFilter adapters
// Takes snake_case API filter, returns CanonicalReportFilter
export function snakeToCamelReportFilter(filter: any): CanonicalReportFilter {
  return {
    // Assuming API filter has 'column', 'operator', 'value'
    column: filter.column,
    operator: filter.operator,
    value: filter.value
  };
}

// Takes CanonicalReportFilter, returns snake_case object for API
export function camelToSnakeReportFilter(filter: CanonicalReportFilter): any {
  return {
    column: filter.column, // Map from CanonicalReportFilter
    operator: filter.operator,
    value: filter.value
  };
}

// ReportSorting adapters
// Takes snake_case API sorting, returns CanonicalReportSorting
export function snakeToCamelReportSorting(sorting: any): CanonicalReportSorting {
  return {
    // Assuming API sorting has 'column', 'direction'
    column: sorting.column,
    direction: sorting.direction
  };
}

// Takes CanonicalReportSorting, returns snake_case object for API
export function camelToSnakeReportSorting(sorting: CanonicalReportSorting): any {
  return {
    column: sorting.column, // Map from CanonicalReportSorting
    direction: sorting.direction
  };
}

// VisualizationConfig adapters
// Takes snake_case API visualization config, returns CanonicalVisualizationConfig
export function snakeToCamelVisualizationConfig(config: any): CanonicalVisualizationConfig {
  // Create a new object with the properties matching CanonicalVisualizationConfig
  return {
    id: config.id || crypto.randomUUID ? crypto.randomUUID() : `viz-${Date.now()}`,
    type: config.type || 'table', // Default to 'table' if not provided
    title: config.title || '',
    dimensions: config.dimensions || [],
    metrics: config.metrics || [],
    settings: config.settings || {}
  };
}

// Takes CanonicalVisualizationConfig, returns snake_case object for API
export function camelToSnakeVisualizationConfig(config: CanonicalVisualizationConfig): any {
  return {
    id: config.id,
    type: config.type || 'table',
    title: config.title,
    dimensions: config.dimensions || [],
    metrics: config.metrics || [],
    settings: config.settings || {}
  };
}

// LayoutItem adapters
// Takes snake_case API layout item, returns CanonicalLayoutItem
export function snakeToCamelLayoutItem(item: any): CanonicalLayoutItem {
  return {
    id: item.id || crypto.randomUUID ? crypto.randomUUID() : `layout-${Date.now()}`,
    x: item.x || 0,
    y: item.y || 0,
    width: item.width || 6, // Default width if not provided
    height: item.height || 4, // Default height if not provided
    content: {
      type: 'visualization',
      visualizationId: item.content?.visualizationId || ''
    }
  };
}

// Takes CanonicalLayoutItem, returns snake_case object for API
export function camelToSnakeLayoutItem(item: CanonicalLayoutItem): any {
  return {
    id: item.id,
    x: item.x,
    y: item.y,
    width: item.width,
    height: item.height,
    content: {
      type: 'visualization',
      visualizationId: item.content.visualizationId
    }
  };
}

// ReportExecutionResult adapters
// Takes snake_case API execution result, returns CanonicalReportExecutionResult
export function snakeToCamelReportExecutionResult(result: any): CanonicalReportExecutionResult {
  // Handle the case when result or result.data might be undefined
  if (!result) {
    return {
      id: '',
      report_id: '',
      timestamp: new Date().toISOString(),
      status: 'error',
      data: {
        columns: [],
        rows: []
      },
      params: {}
    };
  }

  // Return a properly structured CanonicalReportExecutionResult
  return {
    id: result.id || '',
    report_id: result.report_id || '',
    timestamp: result.timestamp || new Date().toISOString(),
    status: result.status || 'success',
    data: {
      columns: result.data?.columns || [],
      rows: result.data?.rows || []
    },
    params: result.params || {},
    error: result.error
  };
}

// Takes CanonicalReportExecutionResult, returns snake_case object for API
export function camelToSnakeReportExecutionResult(result: CanonicalReportExecutionResult): any {
  return {
    id: result.id,
    report_id: result.report_id,
    timestamp: result.timestamp,
    status: result.status,
    data: {
      columns: result.data.columns,
      rows: result.data.rows
    },
    params: result.params,
    error: result.error
  };
}

// Generic adapter for API responses
export function adaptApiResponse<T, U>(
  data: T, 
  adapter: (item: T) => U
): U {
  return adapter(data);
}

// Generic adapter for API response arrays
export function adaptApiResponseArray<T, U>(
  data: T[], 
  adapter: (item: T) => U
): U[] {
  return data.map(item => adapter(item));
}
