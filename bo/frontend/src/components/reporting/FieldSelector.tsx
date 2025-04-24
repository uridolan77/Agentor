import React from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Checkbox,
  Divider,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  ViewColumn as DimensionIcon,
  Functions as MetricIcon,
  Info as InfoIcon
} from '@mui/icons-material';

// Types for the component
export interface Field {
  id: string;
  name: string;
  type: string;
  description?: string;
  table?: string;
}

interface FieldSelectorProps {
  dimensions: Field[];
  metrics: Field[];
  selectedDimensions: string[];
  selectedMetrics: string[];
  onDimensionToggle: (dimensionId: string) => void;
  onMetricToggle: (metricId: string) => void;
}

const FieldSelector: React.FC<FieldSelectorProps> = ({
  dimensions,
  metrics,
  selectedDimensions,
  selectedMetrics,
  onDimensionToggle,
  onMetricToggle
}) => {
  // State for search
  const [dimensionSearch, setDimensionSearch] = React.useState('');
  const [metricSearch, setMetricSearch] = React.useState('');

  // Filter dimensions based on search
  const filteredDimensions = dimensions.filter(dim => 
    dim.name.toLowerCase().includes(dimensionSearch.toLowerCase()) ||
    (dim.table && dim.table.toLowerCase().includes(dimensionSearch.toLowerCase()))
  );

  // Filter metrics based on search
  const filteredMetrics = metrics.filter(metric => 
    metric.name.toLowerCase().includes(metricSearch.toLowerCase()) ||
    (metric.table && metric.table.toLowerCase().includes(metricSearch.toLowerCase()))
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
      {/* Dimensions Panel */}
      <Paper sx={{ flex: 1, maxHeight: 400, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'primary.contrastText' }}>
          <Typography variant="subtitle1" fontWeight="bold">
            Dimensions
          </Typography>
          <Typography variant="caption">
            Select dimensions to group your data
          </Typography>
        </Box>
        
        <Box sx={{ p: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Search dimensions..."
            value={dimensionSearch}
            onChange={(e) => setDimensionSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: dimensionSearch && (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setDimensionSearch('')}
                    edge="end"
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Box>
        
        <Divider />
        
        <List sx={{ overflow: 'auto', flexGrow: 1 }}>
          {filteredDimensions.length === 0 ? (
            <ListItem>
              <ListItemText
                primary="No dimensions found"
                secondary={dimensionSearch ? "Try a different search term" : "No dimensions available"}
              />
            </ListItem>
          ) : (
            filteredDimensions.map((dimension) => (
              <ListItem
                key={dimension.id}
                dense
                button
                onClick={() => onDimensionToggle(dimension.id)}
              >
                <ListItemIcon>
                  <Checkbox
                    edge="start"
                    checked={selectedDimensions.includes(dimension.id)}
                    tabIndex={-1}
                    disableRipple
                  />
                </ListItemIcon>
                <ListItemIcon>
                  <DimensionIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={dimension.name}
                  secondary={dimension.table}
                />
                {dimension.description && (
                  <Tooltip title={dimension.description}>
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </ListItem>
            ))
          )}
        </List>
      </Paper>
      
      {/* Metrics Panel */}
      <Paper sx={{ flex: 1, maxHeight: 400, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, bgcolor: 'secondary.main', color: 'secondary.contrastText' }}>
          <Typography variant="subtitle1" fontWeight="bold">
            Metrics
          </Typography>
          <Typography variant="caption">
            Select metrics to measure your data
          </Typography>
        </Box>
        
        <Box sx={{ p: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Search metrics..."
            value={metricSearch}
            onChange={(e) => setMetricSearch(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: metricSearch && (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setMetricSearch('')}
                    edge="end"
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Box>
        
        <Divider />
        
        <List sx={{ overflow: 'auto', flexGrow: 1 }}>
          {filteredMetrics.length === 0 ? (
            <ListItem>
              <ListItemText
                primary="No metrics found"
                secondary={metricSearch ? "Try a different search term" : "No metrics available"}
              />
            </ListItem>
          ) : (
            filteredMetrics.map((metric) => (
              <ListItem
                key={metric.id}
                dense
                button
                onClick={() => onMetricToggle(metric.id)}
              >
                <ListItemIcon>
                  <Checkbox
                    edge="start"
                    checked={selectedMetrics.includes(metric.id)}
                    tabIndex={-1}
                    disableRipple
                  />
                </ListItemIcon>
                <ListItemIcon>
                  <MetricIcon color="secondary" />
                </ListItemIcon>
                <ListItemText
                  primary={metric.name}
                  secondary={metric.table}
                />
                {metric.description && (
                  <Tooltip title={metric.description}>
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </ListItem>
            ))
          )}
        </List>
      </Paper>
    </Box>
  );
};

export default FieldSelector;
