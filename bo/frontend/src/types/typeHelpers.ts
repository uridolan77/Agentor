/**
 * Type helpers to bridge the gap between snake_case and camelCase type systems
 */

import { 
  DataSource as SnakeCaseDataSource,
  Report as SnakeCaseReport,
  ReportField as SnakeCaseReportField,
  ReportFilter as SnakeCaseReportFilter,
  ReportSorting as SnakeCaseReportSorting,
  VisualizationConfig as SnakeCaseVisualizationConfig,
  LayoutItem as SnakeCaseLayoutItem,
  ReportExecutionResult as SnakeCaseReportExecutionResult
} from './reporting';

import {
  DataSource as CamelCaseDataSource,
  Report as CamelCaseReport,
  ReportField as CamelCaseReportField,
  ReportFilter as CamelCaseReportFilter,
  ReportSorting as CamelCaseReportSorting,
  VisualizationConfig as CamelCaseVisualizationConfig,
  LayoutItem as CamelCaseLayoutItem,
  ReportExecutionResult as CamelCaseReportExecutionResult
} from './reporting/reportTypes';

// Flexible types that can work with both snake_case and camelCase properties
export type FlexibleDataSource = Partial<SnakeCaseDataSource> & Partial<CamelCaseDataSource> & {
  // Additional properties that might be used in the UI but not in the API
  type?: string;
  host?: string;
  port?: string;
  database?: string;
  username?: string;
  password?: string;
  connection_string?: string;
  status?: string;
};

export type FlexibleReport = Partial<SnakeCaseReport> & Partial<CamelCaseReport>;

export type FlexibleReportField = Partial<SnakeCaseReportField> & Partial<CamelCaseReportField>;

export type FlexibleReportFilter = Partial<SnakeCaseReportFilter> & Partial<CamelCaseReportFilter>;

export type FlexibleReportSorting = Partial<SnakeCaseReportSorting> & Partial<CamelCaseReportSorting>;

export type FlexibleVisualizationConfig = Partial<SnakeCaseVisualizationConfig> & Partial<CamelCaseVisualizationConfig>;

export type FlexibleLayoutItem = Partial<SnakeCaseLayoutItem> & Partial<CamelCaseLayoutItem>;

export type FlexibleReportExecutionResult = Partial<SnakeCaseReportExecutionResult> & Partial<CamelCaseReportExecutionResult>;

// Type assertion functions to help TypeScript understand the type conversions
export function asSnakeCaseDataSource(dataSource: FlexibleDataSource): SnakeCaseDataSource {
  return dataSource as unknown as SnakeCaseDataSource;
}

export function asCamelCaseDataSource(dataSource: FlexibleDataSource): CamelCaseDataSource {
  return dataSource as unknown as CamelCaseDataSource;
}

export function asSnakeCaseReport(report: FlexibleReport): SnakeCaseReport {
  return report as unknown as SnakeCaseReport;
}

export function asCamelCaseReport(report: FlexibleReport): CamelCaseReport {
  return report as unknown as CamelCaseReport;
}

// Type guard functions to check if an object has the required properties
export function isSnakeCaseDataSource(dataSource: any): dataSource is SnakeCaseDataSource {
  return (
    typeof dataSource === 'object' &&
    dataSource !== null &&
    'connection_type' in dataSource &&
    'connection_config' in dataSource
  );
}

export function isCamelCaseDataSource(dataSource: any): dataSource is CamelCaseDataSource {
  return (
    typeof dataSource === 'object' &&
    dataSource !== null &&
    'connectionType' in dataSource &&
    'connectionConfig' in dataSource
  );
}

export function isSnakeCaseReport(report: any): report is SnakeCaseReport {
  return (
    typeof report === 'object' &&
    report !== null &&
    'data_source_id' in report &&
    'is_public' in report
  );
}

export function isCamelCaseReport(report: any): report is CamelCaseReport {
  return (
    typeof report === 'object' &&
    report !== null &&
    'dataSourceId' in report &&
    'isPublic' in report
  );
}
