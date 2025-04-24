import React from 'react';
import { Box, Card, CardContent, Typography, Tooltip, Skeleton } from '@mui/material';
import { ChartData, VisualizationConfig } from '../../types/reporting/reportTypes';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon
} from '@mui/icons-material';

interface MetricDisplayProps {
  data: ChartData[];
  config: VisualizationConfig;
  loading?: boolean;
}

const MetricDisplay: React.FC<MetricDisplayProps> = ({ data, config, loading = false }) => {
  // Extract the value from data
  const getValue = () => {
    if (loading || !data || data.length === 0 || !config.measures || config.measures.length === 0) {
      return null;
    }

    const measureId = config.measures[0].field.id;
    const value = data[0][measureId];

    if (typeof value === 'undefined' || value === null) {
      return null;
    }

    return Number(value);
  };

  const value = getValue();

  // Format the value according to the configuration
  const formatValue = (value: number | null) => {
    if (value === null) {
      return 'N/A';
    }

    // Apply formatting based on config
    const format = config.format || {};
    const decimals = format.decimals !== undefined ? format.decimals : 2;

    let formattedValue = value.toFixed(decimals);

    // Add thousands separator if needed
    if (format.thousandsSeparator) {
      const parts = formattedValue.split('.');
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, format.thousandsSeparator);
      formattedValue = parts.join('.');
    }

    // Add prefix and suffix if provided
    if (format.prefix) {
      formattedValue = format.prefix + formattedValue;
    }

    if (format.suffix) {
      formattedValue += format.suffix;
    }

    return formattedValue;
  };

  // Determine trend direction and percentage change
  const getTrend = () => {
    if (!config.trend || !config.trend.compare || value === null) {
      return null;
    }

    const { compare, previousValue } = config.trend;

    if (typeof previousValue !== 'number') {
      return null;
    }

    // Avoid division by zero
    if (previousValue === 0) {
      return {
        direction: value > 0 ? 'up' : value < 0 ? 'down' : 'flat',
        percentage: null
      };
    }

    const change = value - previousValue;
    const percentage = (change / Math.abs(previousValue)) * 100;

    return {
      direction: percentage > 0 ? 'up' : percentage < 0 ? 'down' : 'flat',
      percentage: Math.abs(percentage).toFixed(1)
    };
  };

  const trend = getTrend();

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: config.backgroundColor || 'background.paper'
      }}
    >
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', p: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          {config.title || config.measures?.[0]?.label || 'Metric'}
        </Typography>

        {loading ? (
          <Skeleton variant="rectangular" width="80%" height={60} />
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'baseline' }}>
            <Typography
              variant="h3"
              component="div"
              sx={{
                fontWeight: 'bold',
                color: config.textColor || 'text.primary'
              }}
            >
              {formatValue(value)}
            </Typography>

            {trend && (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  ml: 2,
                  color: trend.direction === 'up'
                    ? 'success.main'
                    : trend.direction === 'down'
                      ? 'error.main'
                      : 'text.secondary'
                }}
              >
                {trend.direction === 'up' ? (
                  <TrendingUpIcon fontSize="small" />
                ) : trend.direction === 'down' ? (
                  <TrendingDownIcon fontSize="small" />
                ) : (
                  <TrendingFlatIcon fontSize="small" />
                )}

                {trend.percentage && (
                  <Typography variant="body2" sx={{ ml: 0.5 }}>
                    {trend.percentage}%
                  </Typography>
                )}
              </Box>
            )}
          </Box>
        )}

        {config.subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {config.subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricDisplay;