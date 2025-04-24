import React, { useMemo } from 'react';
import { styled } from '@mui/material/styles';
import { Box, Typography, Paper, CircularProgress } from '@mui/material';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  PieChart, Pie, ScatterChart, Scatter, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  DataGrid, GridColDef, GridValueGetterParams,
} from '@mui/x-data-grid';

import { ReportElement, VisualizationType } from '../../types/reporting/reportTypes';

// Styled components
const ChartContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  marginBottom: theme.spacing(1),
}));

const ChartSubtitle = styled(Typography)(({ theme }) => ({
  color: theme.palette.text.secondary,
  marginBottom: theme.spacing(2),
}));

const ChartContent = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  minHeight: 0, // Important for flex child to respect container height
}));

const NoDataContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
}));

// Chart colors
const CHART_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

interface ChartRendererProps {
  element: ReportElement;
  data: any[];
  isLoading?: boolean;
  error?: string | null;
}

const ChartRenderer: React.FC<ChartRendererProps> = ({
  element,
  data,
  isLoading = false,
  error = null,
}) => {
  const { visualization } = element;

  // Format data for charts
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    // For simplicity, we'll just use the data as is
    // In a real implementation, you might need to transform the data
    // based on the element's dimensions and metrics
    return data;
  }, [data]);

  // Render loading state
  if (isLoading) {
    return (
      <ChartContainer>
        {visualization.title && <ChartTitle variant="h6">{visualization.title}</ChartTitle>}
        {visualization.subtitle && <ChartSubtitle variant="body2">{visualization.subtitle}</ChartSubtitle>}
        <ChartContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <CircularProgress />
        </ChartContent>
      </ChartContainer>
    );
  }

  // Render error state
  if (error) {
    return (
      <ChartContainer>
        {visualization.title && <ChartTitle variant="h6">{visualization.title}</ChartTitle>}
        {visualization.subtitle && <ChartSubtitle variant="body2">{visualization.subtitle}</ChartSubtitle>}
        <ChartContent>
          <NoDataContainer>
            <Typography color="error" align="center">
              Error: {error}
            </Typography>
          </NoDataContainer>
        </ChartContent>
      </ChartContainer>
    );
  }

  // Render no data state
  if (!chartData || chartData.length === 0) {
    return (
      <ChartContainer>
        {visualization.title && <ChartTitle variant="h6">{visualization.title}</ChartTitle>}
        {visualization.subtitle && <ChartSubtitle variant="body2">{visualization.subtitle}</ChartSubtitle>}
        <ChartContent>
          <NoDataContainer>
            <Typography color="textSecondary" align="center">
              No data available
            </Typography>
          </NoDataContainer>
        </ChartContent>
      </ChartContainer>
    );
  }

  // Render chart based on visualization type
  return (
    <ChartContainer>
      {visualization.title && <ChartTitle variant="h6">{visualization.title}</ChartTitle>}
      {visualization.subtitle && <ChartSubtitle variant="body2">{visualization.subtitle}</ChartSubtitle>}
      <ChartContent>
        {renderChart(visualization.type, chartData, element)}
      </ChartContent>
    </ChartContainer>
  );
};

// Helper function to render the appropriate chart
const renderChart = (type: VisualizationType, data: any[], element: ReportElement) => {
  switch (type) {
    case VisualizationType.TABLE:
      return renderTable(data, element);
    case VisualizationType.BAR_CHART:
      return renderBarChart(data, element);
    case VisualizationType.LINE_CHART:
      return renderLineChart(data, element);
    case VisualizationType.PIE_CHART:
      return renderPieChart(data, element);
    case VisualizationType.AREA_CHART:
      return renderAreaChart(data, element);
    case VisualizationType.SCATTER_CHART:
      return renderScatterChart(data, element);
    case VisualizationType.CARD:
      return renderCard(data, element);
    default:
      return (
        <NoDataContainer>
          <Typography color="textSecondary" align="center">
            Unsupported visualization type: {type}
          </Typography>
        </NoDataContainer>
      );
  }
};

// Render table
const renderTable = (data: any[], element: ReportElement) => {
  // Create columns based on the first data item
  const columns: GridColDef[] = Object.keys(data[0] || {}).map(key => ({
    field: key,
    headerName: key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' '),
    flex: 1,
    minWidth: 120,
  }));

  // Add id to each row if not present
  const rows = data.map((item, index) => ({
    id: item.id || `row-${index}`,
    ...item,
  }));

  return (
    <DataGrid
      rows={rows}
      columns={columns}
      initialState={{
        pagination: {
          paginationModel: { pageSize: 10, page: 0 },
        },
      }}
      pageSizeOptions={[10, 25, 50]}
      disableRowSelectionOnClick
      autoHeight
      density="standard"
      sx={{ border: 'none' }}
    />
  );
};

// Render bar chart
const renderBarChart = (data: any[], element: ReportElement) => {
  // Assume the first dimension is the x-axis
  const xAxisKey = element.dimensions[0] || Object.keys(data[0])[0];

  // Use metrics for the bars
  const metricKeys = element.metrics.length > 0
    ? element.metrics
    : Object.keys(data[0]).filter(key => typeof data[0][key] === 'number');

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {metricKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            fill={CHART_COLORS[index % CHART_COLORS.length]}
            name={key.replace(/_/g, ' ')}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
};

// Render line chart
const renderLineChart = (data: any[], element: ReportElement) => {
  // Assume the first dimension is the x-axis
  const xAxisKey = element.dimensions[0] || Object.keys(data[0])[0];

  // Use metrics for the lines
  const metricKeys = element.metrics.length > 0
    ? element.metrics
    : Object.keys(data[0]).filter(key => typeof data[0][key] === 'number');

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {metricKeys.map((key, index) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={CHART_COLORS[index % CHART_COLORS.length]}
            name={key.replace(/_/g, ' ')}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

// Render pie chart
const renderPieChart = (data: any[], element: ReportElement) => {
  // Assume the first dimension is the name
  const nameKey = element.dimensions[0] || Object.keys(data[0])[0];

  // Use the first metric for the value
  const valueKey = element.metrics[0] || Object.keys(data[0]).find(key => typeof data[0][key] === 'number') || '';

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <Pie
          data={data}
          dataKey={valueKey}
          nameKey={nameKey}
          cx="50%"
          cy="50%"
          outerRadius="80%"
          label={(entry) => entry[nameKey]}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

// Render area chart
const renderAreaChart = (data: any[], element: ReportElement) => {
  // Assume the first dimension is the x-axis
  const xAxisKey = element.dimensions[0] || Object.keys(data[0])[0];

  // Use metrics for the areas
  const metricKeys = element.metrics.length > 0
    ? element.metrics
    : Object.keys(data[0]).filter(key => typeof data[0][key] === 'number');

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {metricKeys.map((key, index) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            fill={CHART_COLORS[index % CHART_COLORS.length]}
            stroke={CHART_COLORS[index % CHART_COLORS.length]}
            fillOpacity={0.3}
            name={key.replace(/_/g, ' ')}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

// Render scatter chart
const renderScatterChart = (data: any[], element: ReportElement) => {
  // Need at least two metrics for scatter chart
  if (element.metrics.length < 2) {
    return (
      <NoDataContainer>
        <Typography color="textSecondary" align="center">
          Scatter chart requires at least two metrics
        </Typography>
      </NoDataContainer>
    );
  }

  const xAxisKey = element.metrics[0];
  const yAxisKey = element.metrics[1];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} name={xAxisKey.replace(/_/g, ' ')} />
        <YAxis dataKey={yAxisKey} name={yAxisKey.replace(/_/g, ' ')} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend />
        <Scatter
          name={`${xAxisKey.replace(/_/g, ' ')} vs ${yAxisKey.replace(/_/g, ' ')}`}
          data={data}
          fill={CHART_COLORS[0]}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
};

// Render card (single value)
const renderCard = (data: any[], element: ReportElement) => {
  // Use the first metric for the value
  const metricKey = element.metrics[0] || Object.keys(data[0]).find(key => typeof data[0][key] === 'number') || '';

  // Calculate the value (e.g., sum, average, etc.)
  let value = 0;
  if (data.length > 0 && metricKey) {
    // For simplicity, just use the first value
    value = data[0][metricKey];
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        padding: 2,
      }}
    >
      <Typography variant="h3" component="div" align="center">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center">
        {metricKey.replace(/_/g, ' ')}
      </Typography>
    </Box>
  );
};

export default ChartRenderer;
