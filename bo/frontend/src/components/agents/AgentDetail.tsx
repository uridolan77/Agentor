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
  Pause as PauseIcon,
  School as TrainingIcon
} from '@mui/icons-material';
import { StatusBadge, LoadingState, ErrorState } from '../common';
import { Agent } from './AgentList';

interface AgentDetailProps {
  agent: Agent | null;
  isLoading: boolean;
  error: string | null;
  onDelete: (id: number) => Promise<void>;
  onToggleActive: (id: number, isActive: boolean) => Promise<void>;
  onRun: (id: number) => Promise<void>;
}

/**
 * AgentDetail component for displaying detailed information about an agent.
 * Includes tabs for overview, configuration, tools, and logs.
 */
const AgentDetail: React.FC<AgentDetailProps> = ({
  agent,
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
    if (agent) {
      navigate(`/agents/${agent.id}/edit`);
    }
  };

  // Handle delete
  const handleDelete = async () => {
    if (agent) {
      await onDelete(agent.id);
      navigate('/agents');
    }
  };

  // Handle toggle active
  const handleToggleActive = async () => {
    if (agent) {
      await onToggleActive(agent.id, !agent.is_active);
    }
  };

  // Handle run
  const handleRun = async () => {
    if (agent) {
      await onRun(agent.id);
    }
  };

  // Handle training
  const handleTraining = () => {
    if (agent) {
      navigate(`/agents/${agent.id}/training`);
    }
  };

  // Render loading state
  if (isLoading) {
    return <LoadingState message="Loading agent details..." />;
  }

  // Render error state
  if (error) {
    return (
      <ErrorState
        title="Error Loading Agent"
        message={error}
        retryLabel="Back to Agents"
        onRetry={() => navigate('/agents')}
      />
    );
  }

  // Render not found state
  if (!agent) {
    return (
      <ErrorState
        title="Agent Not Found"
        message="The requested agent could not be found."
        retryLabel="Back to Agents"
        onRetry={() => navigate('/agents')}
      />
    );
  }

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Parse configuration
  const configuration = typeof agent.configuration === 'string'
    ? JSON.parse(agent.configuration)
    : agent.configuration || {};

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Box display="flex" alignItems="center" mb={1}>
              <Typography variant="h5" component="h1" sx={{ mr: 2 }}>
                {agent.name}
              </Typography>
              <StatusBadge status={agent.is_active ? 'active' : 'inactive'} />
            </Box>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              {agent.description || 'No description provided.'}
            </Typography>
            <Chip 
              label={agent.agent_type} 
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
              disabled={!agent.is_active}
            >
              Run
            </Button>
            <Button
              variant="outlined"
              color={agent.is_active ? 'warning' : 'success'}
              startIcon={agent.is_active ? <PauseIcon /> : <RunIcon />}
              onClick={handleToggleActive}
              sx={{ mr: 1 }}
            >
              {agent.is_active ? 'Deactivate' : 'Activate'}
            </Button>
            <Button
              variant="outlined"
              color="info"
              startIcon={<TrainingIcon />}
              onClick={handleTraining}
              sx={{ mr: 1 }}
            >
              Training
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
          <Tab label="Tools" />
          <Tab label="Logs" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Paper sx={{ p: 3 }}>
        {/* Overview Tab */}
        {tabValue === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Agent Details
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  ID
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {agent.id}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Type
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {agent.agent_type}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Status
                </Typography>
                <Typography variant="body1" gutterBottom>
                  <StatusBadge status={agent.is_active ? 'active' : 'inactive'} />
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Creator ID
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {agent.creator_id}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Created At
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatDate(agent.created_at)}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" color="text.secondary">
                  Last Updated
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {formatDate(agent.updated_at)}
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="text.secondary">
                  Description
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {agent.description || 'No description provided.'}
                </Typography>
              </Grid>
            </Grid>

            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Statistics
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body1" color="text.secondary" align="center" py={4}>
                Agent statistics will be implemented here.
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

        {/* Tools Tab */}
        {tabValue === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Tools
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary" align="center" py={4}>
              Agent tools will be implemented here.
            </Typography>
          </Box>
        )}

        {/* Logs Tab */}
        {tabValue === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Execution Logs
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1" color="text.secondary" align="center" py={4}>
              Agent logs will be implemented here.
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default AgentDetail;
