import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardActions,
  Divider,
  Chip,
  LinearProgress,
  useTheme
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FormField } from '../common';

// Types
interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  validationLoss: number;
  validationAccuracy: number;
  learningRate: number;
  timestamp: string;
}

interface TrainingConfig {
  modelType: string;
  batchSize: number;
  epochs: number;
  learningRate: number;
  optimizer: string;
  datasetId: string;
  validationSplit: number;
  earlyStoppingPatience: number;
  useCheckpointing: boolean;
  checkpointFrequency: number;
}

interface TrainingSession {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'stopped';
  progress: number;
  startTime: string;
  endTime?: string;
  config: TrainingConfig;
  metrics: TrainingMetrics[];
  agentId?: string;
  modelId?: string;
}

interface TrainingDashboardProps {
  agentId?: string;
  onTrainingComplete?: (sessionId: string) => void;
}

/**
 * Training Dashboard Component
 * Provides a UI for configuring, monitoring, and managing agent training sessions
 */
const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ agentId, onTrainingComplete }) => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [activeSession, setActiveSession] = useState<TrainingSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  
  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    modelType: 'transformer',
    batchSize: 32,
    epochs: 10,
    learningRate: 0.001,
    optimizer: 'adam',
    datasetId: '',
    validationSplit: 0.2,
    earlyStoppingPatience: 5,
    useCheckpointing: true,
    checkpointFrequency: 1
  });
  
  // Available datasets and models
  const [availableDatasets, setAvailableDatasets] = useState<{id: string, name: string}[]>([]);
  const [availableModels, setAvailableModels] = useState<{id: string, name: string}[]>([]);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Handle config change
  const handleConfigChange = (name: string, value: any) => {
    setTrainingConfig(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Start training session
  const startTraining = async () => {
    setIsLoading(true);
    try {
      // API call to start training would go here
      const response = await fetch('http://localhost:8000/training/sessions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          agentId,
          config: trainingConfig
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to start training session');
      }
      
      const newSession = await response.json();
      setSessions(prev => [newSession, ...prev]);
      setActiveSession(newSession);
      
      // Connect to WebSocket for real-time updates
      connectWebSocket(newSession.id);
      
    } catch (error) {
      console.error('Error starting training:', error);
      // Handle error
    } finally {
      setIsLoading(false);
    }
  };
  
  // Stop training session
  const stopTraining = async () => {
    if (!activeSession) return;
    
    setIsLoading(true);
    try {
      // API call to stop training would go here
      const response = await fetch(`http://localhost:8000/training/sessions/${activeSession.id}/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop training session');
      }
      
      // Update session status
      const updatedSession = await response.json();
      setSessions(prev => prev.map(session => 
        session.id === updatedSession.id ? updatedSession : session
      ));
      setActiveSession(updatedSession);
      
    } catch (error) {
      console.error('Error stopping training:', error);
      // Handle error
    } finally {
      setIsLoading(false);
    }
  };
  
  // Save trained model
  const saveModel = async () => {
    if (!activeSession) return;
    
    setIsLoading(true);
    try {
      // API call to save model would go here
      const response = await fetch(`http://localhost:8000/training/sessions/${activeSession.id}/save`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to save model');
      }
      
      // Handle successful save
      if (onTrainingComplete) {
        onTrainingComplete(activeSession.id);
      }
      
    } catch (error) {
      console.error('Error saving model:', error);
      // Handle error
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load training sessions
  const loadSessions = async () => {
    setIsLoading(true);
    try {
      // API call to get training sessions would go here
      const response = await fetch(`http://localhost:8000/training/sessions${agentId ? `?agentId=${agentId}` : ''}`);
      
      if (!response.ok) {
        throw new Error('Failed to load training sessions');
      }
      
      const data = await response.json();
      setSessions(data);
      
      // Set active session if there's a running one
      const runningSession = data.find((s: TrainingSession) => s.status === 'running');
      if (runningSession) {
        setActiveSession(runningSession);
        connectWebSocket(runningSession.id);
      }
      
    } catch (error) {
      console.error('Error loading sessions:', error);
      // Handle error
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load available datasets
  const loadDatasets = async () => {
    try {
      // API call to get available datasets would go here
      const response = await fetch('http://localhost:8000/datasets');
      
      if (!response.ok) {
        throw new Error('Failed to load datasets');
      }
      
      const data = await response.json();
      setAvailableDatasets(data);
      
      // Set default dataset if available
      if (data.length > 0) {
        setTrainingConfig(prev => ({
          ...prev,
          datasetId: data[0].id
        }));
      }
      
    } catch (error) {
      console.error('Error loading datasets:', error);
      // Handle error
    }
  };
  
  // Load available models
  const loadModels = async () => {
    try {
      // API call to get available models would go here
      const response = await fetch('http://localhost:8000/models');
      
      if (!response.ok) {
        throw new Error('Failed to load models');
      }
      
      const data = await response.json();
      setAvailableModels(data);
      
    } catch (error) {
      console.error('Error loading models:', error);
      // Handle error
    }
  };
  
  // Connect to WebSocket for real-time updates
  const connectWebSocket = (sessionId: string) => {
    // Close existing connection if any
    if (socket) {
      socket.close();
    }
    
    // Create new WebSocket connection
    const newSocket = new WebSocket(`ws://localhost:8000/training/sessions/${sessionId}/ws`);
    
    newSocket.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Update session with new metrics
      setSessions(prev => prev.map(session => {
        if (session.id === sessionId) {
          return {
            ...session,
            ...data
          };
        }
        return session;
      }));
      
      // Update active session if it's the current one
      if (activeSession && activeSession.id === sessionId) {
        setActiveSession(prev => prev ? { ...prev, ...data } : null);
      }
    };
    
    newSocket.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };
    
    newSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    setSocket(newSocket);
  };
  
  // Load data on component mount
  useEffect(() => {
    loadSessions();
    loadDatasets();
    loadModels();
    
    // Cleanup WebSocket on unmount
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [loadSessions, socket]);
  
  // Render metrics chart
  const renderMetricsChart = () => {
    if (!activeSession || activeSession.metrics.length === 0) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No metrics data available
          </Typography>
        </Box>
      );
    }
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={activeSession.metrics}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
          <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" name="Training Accuracy" />
          <Line type="monotone" dataKey="validationLoss" stroke="#ff7300" name="Validation Loss" />
          <Line type="monotone" dataKey="validationAccuracy" stroke="#0088fe" name="Validation Accuracy" />
        </LineChart>
      </ResponsiveContainer>
    );
  };
  
  // Render session status
  const renderSessionStatus = () => {
    if (!activeSession) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No active training session
          </Typography>
        </Box>
      );
    }
    
    const getStatusColor = (status: string) => {
      switch (status) {
        case 'running':
          return 'primary';
        case 'completed':
          return 'success';
        case 'failed':
          return 'error';
        case 'stopped':
          return 'warning';
        default:
          return 'default';
      }
    };
    
    return (
      <Box sx={{ p: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6">{activeSession.name || 'Training Session'}</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
              <Chip 
                label={activeSession.status.toUpperCase()} 
                color={getStatusColor(activeSession.status) as any}
                size="small"
                sx={{ mr: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Started: {new Date(activeSession.startTime).toLocaleString()}
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={6} sx={{ textAlign: 'right' }}>
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={stopTraining}
              disabled={activeSession.status !== 'running' || isLoading}
              sx={{ mr: 1 }}
            >
              Stop
            </Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={<SaveIcon />}
              onClick={saveModel}
              disabled={activeSession.status !== 'completed' || isLoading}
            >
              Save Model
            </Button>
          </Grid>
          <Grid item xs={12}>
            {activeSession.status === 'running' && (
              <Box sx={{ mt: 1 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={activeSession.progress} 
                  sx={{ height: 10, borderRadius: 5 }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  Progress: {Math.round(activeSession.progress)}% 
                  ({activeSession.metrics.length > 0 ? 
                    `Epoch ${activeSession.metrics[activeSession.metrics.length - 1].epoch}/${activeSession.config.epochs}` : 
                    'Initializing...'})
                </Typography>
              </Box>
            )}
          </Grid>
        </Grid>
      </Box>
    );
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Dashboard" />
          <Tab label="Configuration" />
          <Tab label="History" />
        </Tabs>
        
        {/* Dashboard Tab */}
        {tabValue === 0 && (
          <Box>
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
              {renderSessionStatus()}
            </Box>
            
            <Box sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Training Metrics
              </Typography>
              {renderMetricsChart()}
            </Box>
            
            <Box sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Latest Metrics
              </Typography>
              {activeSession && activeSession.metrics.length > 0 ? (
                <Grid container spacing={2}>
                  {Object.entries(activeSession.metrics[activeSession.metrics.length - 1])
                    .filter(([key]) => key !== 'epoch' && key !== 'timestamp')
                    .map(([key, value]) => (
                      <Grid item xs={6} sm={3} key={key}>
                        <Card variant="outlined">
                          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                            <Typography variant="body2" color="text.secondary">
                              {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                            </Typography>
                            <Typography variant="h6">
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                </Grid>
              ) : (
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
                  No metrics data available
                </Typography>
              )}
            </Box>
          </Box>
        )}
        
        {/* Configuration Tab */}
        {tabValue === 1 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Configuration
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <FormField
                  type="select"
                  name="modelType"
                  label="Model Type"
                  value={trainingConfig.modelType}
                  onChange={handleConfigChange}
                  options={[
                    { value: 'transformer', label: 'Transformer' },
                    { value: 'lstm', label: 'LSTM' },
                    { value: 'gpt', label: 'GPT-like' },
                    { value: 'bert', label: 'BERT-like' }
                  ]}
                  helperText="Type of model architecture to use"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="select"
                  name="datasetId"
                  label="Dataset"
                  value={trainingConfig.datasetId}
                  onChange={handleConfigChange}
                  options={availableDatasets.map(dataset => ({
                    value: dataset.id,
                    label: dataset.name
                  }))}
                  helperText="Dataset to use for training"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="batchSize"
                  label="Batch Size"
                  value={trainingConfig.batchSize}
                  onChange={handleConfigChange}
                  min={1}
                  max={512}
                  helperText="Number of samples per batch"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="epochs"
                  label="Epochs"
                  value={trainingConfig.epochs}
                  onChange={handleConfigChange}
                  min={1}
                  max={1000}
                  helperText="Number of complete passes through the dataset"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="learningRate"
                  label="Learning Rate"
                  value={trainingConfig.learningRate}
                  onChange={handleConfigChange}
                  min={0.00001}
                  max={1}
                  step={0.0001}
                  helperText="Step size for gradient updates"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="select"
                  name="optimizer"
                  label="Optimizer"
                  value={trainingConfig.optimizer}
                  onChange={handleConfigChange}
                  options={[
                    { value: 'adam', label: 'Adam' },
                    { value: 'sgd', label: 'SGD' },
                    { value: 'rmsprop', label: 'RMSprop' },
                    { value: 'adagrad', label: 'Adagrad' }
                  ]}
                  helperText="Optimization algorithm"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="validationSplit"
                  label="Validation Split"
                  value={trainingConfig.validationSplit}
                  onChange={handleConfigChange}
                  min={0}
                  max={0.5}
                  step={0.01}
                  helperText="Fraction of data to use for validation"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="earlyStoppingPatience"
                  label="Early Stopping Patience"
                  value={trainingConfig.earlyStoppingPatience}
                  onChange={handleConfigChange}
                  min={0}
                  max={100}
                  helperText="Number of epochs with no improvement before stopping"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="switch"
                  name="useCheckpointing"
                  label="Use Checkpointing"
                  value={trainingConfig.useCheckpointing}
                  onChange={handleConfigChange}
                  helperText="Save model checkpoints during training"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormField
                  type="number"
                  name="checkpointFrequency"
                  label="Checkpoint Frequency"
                  value={trainingConfig.checkpointFrequency}
                  onChange={handleConfigChange}
                  min={1}
                  max={100}
                  disabled={!trainingConfig.useCheckpointing}
                  helperText="Save checkpoint every N epochs"
                />
              </Grid>
              
              <Grid item xs={12} sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<StartIcon />}
                  onClick={startTraining}
                  disabled={isLoading || !trainingConfig.datasetId}
                  fullWidth
                >
                  {isLoading ? <CircularProgress size={24} /> : 'Start Training'}
                </Button>
              </Grid>
            </Grid>
          </Box>
        )}
        
        {/* History Tab */}
        {tabValue === 2 && (
          <Box sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Training History
              </Typography>
              <Button
                startIcon={<RefreshIcon />}
                onClick={loadSessions}
                disabled={isLoading}
              >
                Refresh
              </Button>
            </Box>
            
            {isLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : sessions.length > 0 ? (
              <Grid container spacing={2}>
                {sessions.map(session => (
                  <Grid item xs={12} key={session.id}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="h6">
                            {session.name || `Session ${session.id.substring(0, 8)}`}
                          </Typography>
                          <Chip 
                            label={session.status.toUpperCase()} 
                            color={
                              session.status === 'completed' ? 'success' :
                              session.status === 'running' ? 'primary' :
                              session.status === 'failed' ? 'error' :
                              session.status === 'stopped' ? 'warning' : 'default'
                            }
                            size="small"
                          />
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Started: {new Date(session.startTime).toLocaleString()}
                          {session.endTime && ` • Ended: ${new Date(session.endTime).toLocaleString()}`}
                        </Typography>
                        
                        <Divider sx={{ my: 1.5 }} />
                        
                        <Grid container spacing={1}>
                          <Grid item xs={6} sm={3}>
                            <Typography variant="body2" color="text.secondary">Model Type</Typography>
                            <Typography variant="body1">{session.config.modelType}</Typography>
                          </Grid>
                          <Grid item xs={6} sm={3}>
                            <Typography variant="body2" color="text.secondary">Epochs</Typography>
                            <Typography variant="body1">
                              {session.metrics.length > 0 ? 
                                `${session.metrics[session.metrics.length - 1].epoch}/${session.config.epochs}` : 
                                `0/${session.config.epochs}`}
                            </Typography>
                          </Grid>
                          <Grid item xs={6} sm={3}>
                            <Typography variant="body2" color="text.secondary">Batch Size</Typography>
                            <Typography variant="body1">{session.config.batchSize}</Typography>
                          </Grid>
                          <Grid item xs={6} sm={3}>
                            <Typography variant="body2" color="text.secondary">Learning Rate</Typography>
                            <Typography variant="body1">{session.config.learningRate}</Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                      <CardActions>
                        <Button 
                          size="small" 
                          onClick={() => setActiveSession(session)}
                          disabled={activeSession?.id === session.id}
                        >
                          View Details
                        </Button>
                        {session.status === 'completed' && (
                          <Button 
                            size="small" 
                            color="primary"
                            onClick={() => {
                              setActiveSession(session);
                              saveModel();
                            }}
                          >
                            Use Model
                          </Button>
                        )}
                      </CardActions>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body1" color="text.secondary">
                  No training sessions found
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default TrainingDashboard;
