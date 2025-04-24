import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Tooltip,
  Chip,
  useTheme
} from '@mui/material';
import { 
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  ContentCopy as DuplicateIcon,
  MoreVert as MoreIcon
} from '@mui/icons-material';
import { Link, useNavigate } from 'react-router-dom';
import { PageHeader, EmptyState, LoadingState, ErrorState, SearchBar, ConfirmDialog } from '../../components/common';

// Mock workflow data
interface Workflow {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'inactive' | 'draft';
  lastRun?: string;
  createdAt: string;
  updatedAt: string;
  nodeCount: number;
}

const mockWorkflows: Workflow[] = [
  // Empty for now to show the empty state
];

const WorkflowsPage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [workflows, setWorkflows] = useState<Workflow[]>(mockWorkflows);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);

  // Fetch workflows
  const fetchWorkflows = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('/api/workflows');
      // setWorkflows(response.data);
      
      // Using mock data for now
      setWorkflows(mockWorkflows);
    } catch (err: any) {
      console.error('Error fetching workflows:', err);
      setError('Failed to load workflows. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load workflows on component mount
  useEffect(() => {
    fetchWorkflows();
  }, [fetchWorkflows]);

  // Handle search
  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  // Filter workflows based on search query
  const filteredWorkflows = workflows.filter(workflow => 
    workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    workflow.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Handle create workflow
  const handleCreateWorkflow = () => {
    navigate('/workflows/create');
  };

  // Handle delete workflow
  const handleDeleteWorkflow = (workflow: Workflow) => {
    setSelectedWorkflow(workflow);
    setDeleteDialogOpen(true);
  };

  // Handle delete confirm
  const handleDeleteConfirm = async () => {
    if (!selectedWorkflow) return;
    
    setIsLoading(true);
    
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/workflows/${selectedWorkflow.id}`);
      
      // Update state
      setWorkflows(workflows.filter(w => w.id !== selectedWorkflow.id));
      
      // Show success message
      console.log('Workflow deleted successfully');
    } catch (err: any) {
      console.error('Error deleting workflow:', err);
      setError('Failed to delete workflow. Please try again.');
    } finally {
      setIsLoading(false);
      setDeleteDialogOpen(false);
      setSelectedWorkflow(null);
    }
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return theme.palette.success.main;
      case 'inactive':
        return theme.palette.error.main;
      case 'draft':
        return theme.palette.warning.main;
      default:
        return theme.palette.info.main;
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Workflows"
        breadcrumbs={[{ label: 'Workflows' }]}
      />
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <SearchBar 
          placeholder="Search workflows..."
          onSearch={handleSearch}
        />
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={handleCreateWorkflow}
        >
          Create Workflow
        </Button>
      </Box>
      
      {isLoading ? (
        <LoadingState message="Loading workflows..." />
      ) : error ? (
        <ErrorState message={error} onRetry={fetchWorkflows} />
      ) : filteredWorkflows.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <EmptyState
            title="No workflows found"
            message="Workflow management interface will be implemented here."
            icon="workflow"
            actionLabel="Create Workflow"
            onAction={handleCreateWorkflow}
          />
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {filteredWorkflows.map(workflow => (
            <Grid item xs={12} md={6} lg={4} key={workflow.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="h6" component="div" noWrap>
                      {workflow.name}
                    </Typography>
                    <Chip
                      label={workflow.status.toUpperCase()}
                      size="small"
                      sx={{
                        bgcolor: `${getStatusColor(workflow.status)}20`,
                        color: getStatusColor(workflow.status),
                        fontWeight: 'bold'
                      }}
                    />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {workflow.description || 'No description'}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Nodes: {workflow.nodeCount}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Last Updated: {formatDate(workflow.updatedAt)}
                    </Typography>
                  </Box>
                  
                  {workflow.lastRun && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Last Run: {formatDate(workflow.lastRun)}
                    </Typography>
                  )}
                </CardContent>
                
                <CardActions sx={{ justifyContent: 'space-between' }}>
                  <Button
                    component={Link}
                    to={`/workflows/${workflow.id}`}
                    size="small"
                    color="primary"
                  >
                    View Details
                  </Button>
                  
                  <Box>
                    <Tooltip title="Run">
                      <IconButton size="small" color="success">
                        <PlayArrowIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton 
                        size="small" 
                        component={Link} 
                        to={`/workflows/${workflow.id}/edit`}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Duplicate">
                      <IconButton size="small">
                        <DuplicateIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={() => handleDeleteWorkflow(workflow)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
      
      <ConfirmDialog
        open={deleteDialogOpen}
        title="Delete Workflow"
        message={
          selectedWorkflow
            ? `Are you sure you want to delete the workflow "${selectedWorkflow.name}"? This action cannot be undone.`
            : 'Are you sure you want to delete this workflow? This action cannot be undone.'
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteDialogOpen(false)}
        isLoading={isLoading}
        confirmColor="error"
      />
    </Box>
  );
};

export default WorkflowsPage;
