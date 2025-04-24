import React, { useMemo } from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { ChartData, VisualizationConfig } from '../../types/reporting/reportTypes';

interface PieChartProps {
  data: ChartData[];
  config: VisualizationConfig;
}

const PieChart: React.FC<PieChartProps> = ({ data, config }) => {
  const theme = useTheme();

  // Transform data for pie chart display
  const pieData = useMemo(() => {
    if (!data || data.length === 0 || !config.measures || config.measures.length === 0) {
      return [];
    }

    // Use the first measure for pie chart
    const measureId = config.measures[0].field.id;
    const dimensionId = config.dimension?.field?.id || 'name';

    return data.map(item => ({
      name: item[dimensionId]?.toString() || 'Unknown',
      value: Number(item[measureId]) || 0
    }));
  }, [data, config.measures, config.dimension]);

  // Generate colors for pie segments
  const colors = useMemo(() => {
    const defaultColors = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.error.main,
      theme.palette.warning.main,
      theme.palette.info.main,
      theme.palette.success.main,
      theme.palette.primary.light,
      theme.palette.secondary.light,
      theme.palette.error.light,
      theme.palette.warning.light,
      theme.palette.info.light,
      theme.palette.success.light,
      theme.palette.primary.dark,
      theme.palette.secondary.dark,
      theme.palette.error.dark,
      theme.palette.warning.dark,
    ];

    // If custom colors are provided in config, use them
    if (config.colors && config.colors.length > 0) {
      return config.colors;
    }

    return defaultColors;
  }, [config.colors, theme.palette]);

  // Handle empty data state
  if (!pieData || pieData.length === 0) {
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
        <RechartsPieChart>
          <Pie
            data={pieData}
            cx="50%"
            cy="50%"
            labelLine={config.chartSubtype !== 'compact'}
            outerRadius={config.chartSubtype === 'compact' ? 80 : 110}
            innerRadius={config.chartSubtype === 'donut' ? 60 : 0}
            fill="#8884d8"
            dataKey="value"
            label={config.chartSubtype !== 'compact' ?
              ({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%` : false}
          >
            {pieData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={colors[index % colors.length]}
              />
            ))}
          </Pie>
          <Tooltip
            formatter={(value) => [`${value}`, config.measures?.[0]?.label || 'Value']}
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              borderColor: theme.palette.divider,
              color: theme.palette.text.primary,
              borderRadius: 4,
              boxShadow: theme.shadows[3]
            }}
          />
          <Legend
            layout="horizontal"
            verticalAlign="bottom"
            wrapperStyle={{ paddingTop: 20 }}
          />
        </RechartsPieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PieChart;