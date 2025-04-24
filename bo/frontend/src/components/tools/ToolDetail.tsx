import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Grid,
  Button,
  Chip,
  IconButton,
  Tooltip,
  Tab,
  Tabs
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as RunIcon,
  Pause as PauseIcon
} from '@mui/icons-material';
import { StatusBadge, LoadingState, ErrorState } from '../common';
import { Tool } from './ToolList';

interface ToolDetailProps {
  tool: Tool | null;
  isLoading: boolean;
  error: string | null;
  onDelete: (id: number) => Promise<void>;
  onToggleActive: (id: number, isActive: boolean) => Promise<void>;
  onRun: (id: number) => Promise<void>;
}

/**
 * ToolDetail component for displaying detailed information about a tool.
 * Includes tabs for overview, configuration, schema, and usage.
 */
const ToolDetail: React.FC<ToolDetailProps> = ({
  tool,
  isLoading,
  error,
  onDelete,
  onToggleActive,
  onRun
}) => {
  const navigate = useNavigate();
  const [tabValue, setTabValue] = React.useState(0);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle edit
  const handleEdit = () => {
    if (tool) {
      navigate(`/tools/${tool.id}/edit`);
    }
  };

  // Handle delete
  const handleDelete = async () => {
    if (tool) {
      await onDelete(tool.id);
      navigate('/tools');
    }
  };

  // Handle toggle active
  const handleToggleActive = async () => {
    if (tool) {
      await onToggleActive(tool.id, !tool.is_active);
    }
  };

  // Handle run
  const handleRun = async () => {
    if (tool) {
      await onRun(tool.id);
    }
  };

  // Render loading state
  if (isLoading) {
    return <LoadingState message="Loading tool details..." />;
  }

  // Render error state
  if (error) {
    return (
      <ErrorState
        title="Error Loading Tool"
        message={error}
        retryLabel="Back to Tools"
        onRetry={() => navigate('/tools')}
      />
    );
  }

  // Render not found state
  if (!tool) {
    return (
      <ErrorState
        title="Tool Not Found"
        message="The requested tool could not be found."
        retryLabel="Back to Tools"
        onRetry={() => navigate('/tools')}
      />
    );
  }

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Parse configuration
  const configuration = typeof tool.configuration === 'string'
    ? JSON.parse(tool.configuration)
    : tool.configuration || {};

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Box display="flex" alignItems="center" mb={1}>
              <Typography variant="h5" component="h1" sx={{ mr: 2 }}>
                {tool.name}
              </Typography>
              <StatusBadge status={tool.is_active ? 'active' : 'inactive'} />
            </Box>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              {tool.description || 'No description provided.'}
            </Typography>
            <Chip 
              label={tool.tool_type} 
              size="small" 
              sx={{ mt: 1 }} 
            />
          </Box>
          <Box>
            <Button
              variant="contained"
              color="primary"
              startIcon={<RunIcon />}
              onClick={handleRun}
              sx={{ mr: 1 }}
              disabled={!tool.is_active}
            >
              Run
            </Button>
            <Button
              variant="outlined"
              color={tool.is_active ? 'warning' : 'success'}
              startIcon={tool.is_active ? <PauseIcon /> : <RunIcon />}
              onClick={handleToggleActive}
              sx={{ mr: 1 }}
            >
              {tool.is_active ? 'Deactivate' : 'Activate'}
            </Button>
            <Tooltip title="Edit">
              <IconButton onClick={handleEdit} sx={{ mr: 1 }}>
                <EditIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Delete">
              <IconButton color="error" onClick={handleDelete}>
                <DeleteIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Overview" />
          <Tab label="Configuration" />
          <Tab label="Schema" />
          <Tab label="Usage" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Paper sx={{ p: 3 }}>
        {/* Overview Tab */}
        {tabValue === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Tool Details
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  ID
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {tool.id}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Type
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {tool.tool_type}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Status
                </Typography>
                <Typography variant="body1" gutterBottom>
                  <StatusBadge status={tool.is_active ? 'active' : 'inactive'} />
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Creator ID
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {tool.creator_id}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Created At
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatDate(tool.created_at)}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Last Updated
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatDate(tool.updated_at)}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="text.secondary">
                  Description
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {tool.description || 'No description provided.'}
                </Typography>
              </Grid>
            </Grid>

            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Statistics
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body1" color="text.secondary" align="center" py={4}>
                Tool statistics will be implemented here.
              </Typography>
            </Box>
          </Box>
        )}

        {/* Configuration Tab */}
        {tabValue === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {Object.keys(configuration).length > 0 ? (
              <Grid container spacing={2}>
                {Object.entries(configuration).map(([key, value]) => (
                  <Grid item xs={12} key={key}>
                    <Typography variant="subtitle2" color="text.secondary">
                      {key}
                    </Typography>
                    <Typography variant="body1" gutterBottom>
                      {typeof value === 'object' 
                        ? JSON.stringify(value, null, 2) 
                        : String(value)}
                    </Typography>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography variant="body1" color="text.secondary" align="center" py={4}>
                No configuration settings found.
              </Typography>
            )}
          </Box>
        )}

        {/* Schema Tab */}
        {tabValue === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Input/Output Schema
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary" align="center" py={4}>
              Tool schema will be implemented here.
            </Typography>
          </Box>
        )}

        {/* Usage Tab */}
        {tabValue === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Usage History
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary" align="center" py={4}>
              Tool usage history will be implemented here.
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ToolDetail;
