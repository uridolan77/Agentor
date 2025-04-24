import React, { useMemo } from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  TooltipProps
} from 'recharts';
import { ChartData, VisualizationConfig } from '../../types/reporting/reportTypes';

interface BarChartProps {
  data: ChartData[];
  config: VisualizationConfig;
}

const BarChart: React.FC<BarChartProps> = ({ data, config }) => {
  const theme = useTheme();

  // Generate bars for each measure in the config
  const bars = useMemo(() => {
    if (!config.measures || config.measures.length === 0) {
      return [];
    }

    return config.measures.map((measure, index) => {
      const colors = [
        theme.palette.primary.main,
        theme.palette.secondary.main,
        theme.palette.error.main,
        theme.palette.warning.main,
        theme.palette.info.main,
        theme.palette.success.main,
      ];

      return (
        <Bar
          key={measure.field.id}
          dataKey={measure.field.id}
          name={measure.label || measure.field.name}
          fill={measure.color || colors[index % colors.length]}
          stackId={config.chartSubtype === 'stacked' ? '1' : undefined}
        />
      );
    });
  }, [config.measures, config.chartSubtype, theme.palette]);

  // Handle empty data state
  if (!data || data.length === 0) {
    return (
      <Box
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          p: 2
        }}
      >
        <Typography variant="body1" color="text.secondary">
          No data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <RechartsBarChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 70
          }}
          barSize={config.chartSubtype === 'stacked' ? undefined : 20}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis
            dataKey={config.dimension?.field?.id || 'name'}
            angle={-45}
            textAnchor="end"
            height={70}
            tick={{ fill: theme.palette.text.primary, fontSize: 12 }}
            tickLine={{ stroke: theme.palette.divider }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <YAxis
            tick={{ fill: theme.palette.text.primary, fontSize: 12 }}
            tickLine={{ stroke: theme.palette.divider }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              borderColor: theme.palette.divider,
              color: theme.palette.text.primary,
              borderRadius: 4,
              boxShadow: theme.shadows[3]
            }}
          />
          <Legend
            wrapperStyle={{
              paddingTop: 10,
              color: theme.palette.text.primary
            }}
          />
          {bars}
        </RechartsBarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default BarChart;