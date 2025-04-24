import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Tabs,
  Tab,
  Grid,
  useTheme
} from '@mui/material';
import { 
  ArrowBack as ArrowBackIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  Save as SaveIcon
} from '@mui/icons-material';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { PageHeader } from '../../components/common';
import { 
  WorkflowExecutionPanel,
  ExecutionLog,
  ExecutionStatus
} from '../../components/workflows';
import { 
  WorkflowCanvas,
  NodeData,
  EdgeData,
  NodeType,
  EdgeType
} from '../../components/workflows/canvas';
import { Node, Edge } from 'reactflow';

// Define WorkflowData interface
interface WorkflowData {
  nodes: Node<NodeData>[];
  edges: Edge<EdgeData>[];
}

// Mock workflow data
const mockWorkflow: WorkflowData = {
  nodes: [
    {
      id: 'node-1',
      type: NodeType.AGENT,
      position: { x: 100, y: 100 },
      data: {
        id: 'node-1',
        type: NodeType.AGENT,
        label: 'Agent 1',
        description: 'Processes input data',
        position: { x: 100, y: 100 },
        config: {}
      }
    },
    {
      id: 'node-2',
      type: NodeType.TOOL,
      position: { x: 400, y: 100 },
      data: {
        id: 'node-2',
        type: NodeType.TOOL,
        label: 'Tool 1',
        description: 'Transforms data',
        position: { x: 400, y: 100 },
        config: {}
      }
    },
    {
      id: 'node-3',
      type: NodeType.CONDITION,
      position: { x: 700, y: 100 },
      data: {
        id: 'node-3',
        type: NodeType.CONDITION,
        label: 'Condition 1',
        description: 'Checks data validity',
        position: { x: 700, y: 100 },
        config: {}
      }
    },
    {
      id: 'node-4',
      type: NodeType.OUTPUT,
      position: { x: 1000, y: 100 },
      data: {
        id: 'node-4',
        type: NodeType.OUTPUT,
        label: 'Output 1',
        description: 'Final result',
        position: { x: 1000, y: 100 },
        config: {}
      }
    }
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      data: {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        sourceHandle: '',
        targetHandle: '',
        type: EdgeType.DATA,
        label: 'Data Flow'
      }
    },
    {
      id: 'edge-2',
      source: 'node-2',
      target: 'node-3',
      data: {
        id: 'edge-2',
        source: 'node-2',
        target: 'node-3',
        sourceHandle: '',
        targetHandle: '',
        type: EdgeType.DATA,
        label: 'Data Flow'
      }
    },
    {
      id: 'edge-3',
      source: 'node-3',
      target: 'node-4',
      data: {
        id: 'edge-3',
        source: 'node-3',
        target: 'node-4',
        sourceHandle: '',
        targetHandle: '',
        type: EdgeType.CONTROL,
        label: 'Control Flow'
      }
    }
  ]
};

// Mock execution logs
const mockLogs: ExecutionLog[] = [
  {
    id: 'log-1',
    timestamp: new Date().toISOString(),
    level: 'info',
    message: 'Workflow execution started'
  },
  {
    id: 'log-2',
    timestamp: new Date().toISOString(),
    level: 'info',
    nodeId: 'node-1',
    nodeName: 'Agent 1',
    message: 'Processing input data'
  },
  {
    id: 'log-3',
    timestamp: new Date().toISOString(),
    level: 'success',
    nodeId: 'node-1',
    nodeName: 'Agent 1',
    message: 'Input data processed successfully'
  },
  {
    id: 'log-4',
    timestamp: new Date().toISOString(),
    level: 'info',
    nodeId: 'node-2',
    nodeName: 'Tool 1',
    message: 'Transforming data'
  },
  {
    id: 'log-5',
    timestamp: new Date().toISOString(),
    level: 'warning',
    nodeId: 'node-2',
    nodeName: 'Tool 1',
    message: 'Data transformation warning',
    details: 'Some fields could not be transformed properly'
  }
];

// Mock execution status
const mockStatus: ExecutionStatus = {
  status: 'completed',
  progress: 100,
  startTime: new Date(Date.now() - 60000).toISOString(),
  endTime: new Date().toISOString()
};

const WorkflowDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [workflow, setWorkflow] = useState<WorkflowData>(mockWorkflow);
  const [executionLogs, setExecutionLogs] = useState<ExecutionLog[]>(mockLogs);
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus>(mockStatus);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch workflow data
  const fetchWorkflowData = useCallback(async () => {
    if (!id) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`/api/workflows/${id}`);
      // setWorkflow(response.data);
      
      // Using mock data for now
      setWorkflow(mockWorkflow);
    } catch (err: any) {
      console.error('Error fetching workflow data:', err);
      setError('Failed to load workflow data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  // Load workflow data on component mount
  useEffect(() => {
    fetchWorkflowData();
  }, [fetchWorkflowData]);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle workflow change
  const handleWorkflowChange = (updatedWorkflow: WorkflowData) => {
    setWorkflow(updatedWorkflow);
  };

  // Handle workflow run
  const handleRunWorkflow = () => {
    // In a real app, this would be an API call
    // await axios.post(`/api/workflows/${id}/run`);
    
    // Update execution status
    setExecutionStatus({
      status: 'running',
      progress: 0,
      startTime: new Date().toISOString()
    });
    
    // Clear logs
    setExecutionLogs([{
      id: `log-${Date.now()}`,
      timestamp: new Date().toISOString(),
      level: 'info',
      message: 'Workflow execution started'
    }]);
    
    // Simulate workflow execution
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      
      // Add a log entry
      setExecutionLogs(prev => [
        ...prev,
        {
          id: `log-${Date.now()}`,
          timestamp: new Date().toISOString(),
          level: Math.random() > 0.8 ? 'warning' : 'info',
          nodeId: `node-${Math.floor(Math.random() * 4) + 1}`,
          nodeName: `Node ${Math.floor(Math.random() * 4) + 1}`,
          message: `Execution progress: ${progress}%`
        }
      ]);
      
      // Update progress
      setExecutionStatus(prev => ({
        ...prev,
        progress,
        status: progress < 100 ? 'running' : 'completed',
        endTime: progress >= 100 ? new Date().toISOString() : undefined
      }));
      
      if (progress >= 100) {
        clearInterval(interval);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  };

  // Handle workflow stop
  const handleStopWorkflow = () => {
    // In a real app, this would be an API call
    // await axios.post(`/api/workflows/${id}/stop`);
    
    // Update execution status
    setExecutionStatus(prev => ({
      ...prev,
      status: 'stopped',
      endTime: new Date().toISOString()
    }));
    
    // Add a log entry
    setExecutionLogs(prev => [
      ...prev,
      {
        id: `log-${Date.now()}`,
        timestamp: new Date().toISOString(),
        level: 'warning',
        message: 'Workflow execution stopped by user'
      }
    ]);
  };

  // Handle clear logs
  const handleClearLogs = () => {
    setExecutionLogs([]);
  };

  // Handle save execution
  const handleSaveExecution = () => {
    // In a real app, this would be an API call
    // await axios.post(`/api/workflows/${id}/executions`, {
    //   logs: executionLogs,
    //   status: executionStatus
    // });
    
    console.log('Execution saved');
  };

  // Handle download logs
  const handleDownloadLogs = () => {
    const logsJson = JSON.stringify(executionLogs, null, 2);
    const blob = new Blob([logsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow-${id}-logs.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <PageHeader 
        title="Workflow Details"
        breadcrumbs={[
          { label: 'Workflows', path: '/workflows' },
          { label: 'Workflow Details' }
        ]}
      />
      
      {/* Actions */}
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'flex-end', 
          mb: 2 
        }}
      >
        <Button
          variant="contained"
          color="success"
          startIcon={<PlayArrowIcon />}
          onClick={handleRunWorkflow}
          disabled={executionStatus.status === 'running'}
          sx={{ mr: 1 }}
        >
          Run
        </Button>
        <Button
          variant="outlined"
          color="primary"
          startIcon={<SaveIcon />}
          sx={{ mr: 1 }}
        >
          Save
        </Button>
        <Button
          variant="outlined"
          color="primary"
          startIcon={<EditIcon />}
          sx={{ mr: 1 }}
        >
          Edit
        </Button>
        <Button
          variant="outlined"
          color="error"
          startIcon={<DeleteIcon />}
        >
          Delete
        </Button>
      </Box>
      
      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Designer" />
          <Tab label="Overview" />
          <Tab label="Executions" />
          <Tab label="Configuration" />
        </Tabs>
      </Paper>
      
      {/* Tab Content */}
      {tabValue === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Paper 
              sx={{ 
                height: 600, 
                mb: 2 
              }}
            >
              <WorkflowCanvas
                initialNodes={workflow.nodes}
                initialEdges={workflow.edges}
                onNodesChange={(nodes) => {
                  handleWorkflowChange({
                    ...workflow,
                    nodes
                  });
                }}
                onEdgesChange={(edges) => {
                  handleWorkflowChange({
                    ...workflow,
                    edges
                  });
                }}
                onSave={(nodes, edges) => {
                  handleWorkflowChange({
                    nodes,
                    edges
                  });
                }}
              />
            </Paper>
          </Grid>
          <Grid item xs={12}>
            <WorkflowExecutionPanel
              workflowId={id || ''}
              workflowName="Sample Workflow"
              status={executionStatus}
              logs={executionLogs}
              onStart={handleRunWorkflow}
              onStop={handleStopWorkflow}
              onClearLogs={handleClearLogs}
              onSaveExecution={handleSaveExecution}
              onDownloadLogs={handleDownloadLogs}
            />
          </Grid>
        </Grid>
      )}
      
      {tabValue === 1 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Overview</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Name
              </Typography>
              <Typography variant="body1" gutterBottom>
                Sample Workflow
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                ID
              </Typography>
              <Typography variant="body1" gutterBottom>
                {id}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Created
              </Typography>
              <Typography variant="body1" gutterBottom>
                April 20, 2025
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="text.secondary">
                Last Modified
              </Typography>
              <Typography variant="body1" gutterBottom>
                April 21, 2025
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2" color="text.secondary">
                Description
              </Typography>
              <Typography variant="body1" gutterBottom>
                This is a sample workflow that demonstrates the capabilities of the workflow editor.
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}
      
      {tabValue === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Executions</Typography>
          <Typography variant="body1" color="text.secondary" align="center" py={8}>
            Workflow execution history will be implemented here.
          </Typography>
        </Paper>
      )}
      
      {tabValue === 3 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Configuration</Typography>
          <Typography variant="body1" color="text.secondary" align="center" py={8}>
            Workflow configuration will be implemented here.
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default WorkflowDetailPage;
