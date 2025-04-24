import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  useTheme
} from '@mui/material';
import {
  SmartToy as AgentIcon,
  Build as ToolIcon,
  AccountTree as WorkflowIcon,
  Cloud as LLMIcon,
  Person as UserIcon
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import axios from 'axios';

// Mock data for initial development
const mockData = {
  stats: {
    agents: 24,
    tools: 56,
    workflows: 12,
    llmConnections: 8,
    users: 15
  },
  recentAgents: [
    { id: 1, name: 'Customer Support Bot', type: 'MemoryEnhancedAgent', status: 'Active' },
    { id: 2, name: 'Data Processor', type: 'UtilityBasedAgent', status: 'Active' },
    { id: 3, name: 'Content Moderator', type: 'RuleBasedAgent', status: 'Inactive' }
  ],
  recentTools: [
    { id: 1, name: 'Weather API', type: 'EnhancedTool', usage: 'High' },
    { id: 2, name: 'Data Analyzer', type: 'ComposableTool', usage: 'Medium' },
    { id: 3, name: 'PDF Extractor', type: 'EnhancedTool', usage: 'Low' }
  ],
  systemLogs: [
    { level: 'INFO', message: 'AgentRegistry startup complete', time: '5m ago' },
    { level: 'WARNING', message: 'Rate limit reached for OpenAI API', time: '23m ago' },
    { level: 'ERROR', message: 'Failed to connect to database', time: '1h ago' }
  ]
};

// Stat card component
interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color }) => {
  const theme = useTheme();
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ display: 'flex', alignItems: 'center' }}>
        <Box 
          sx={{ 
            bgcolor: color, 
            color: 'white', 
            p: 1.5, 
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mr: 2
          }}
        >
          {icon}
        </Box>
        <Box>
          <Typography variant="h4" component="div">
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {title}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const { user } = useAuth();
  const [data, setData] = useState(mockData);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch dashboard data
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // In a real app, this would be an API call
        // const response = await axios.get('/api/dashboard');
        // setData(response.data);
        
        // Using mock data for now
        setData(mockData);
      } catch (err: any) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, []);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Welcome back, {user?.username}!
      </Typography>

      {/* Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={4} lg={2.4}>
          <StatCard 
            title="Agents" 
            value={data.stats.agents} 
            icon={<AgentIcon />} 
            color={theme.palette.primary.main} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2.4}>
          <StatCard 
            title="Tools" 
            value={data.stats.tools} 
            icon={<ToolIcon />} 
            color={theme.palette.secondary.main} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2.4}>
          <StatCard 
            title="Workflows" 
            value={data.stats.workflows} 
            icon={<WorkflowIcon />} 
            color={theme.palette.success.main} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2.4}>
          <StatCard 
            title="LLM Connections" 
            value={data.stats.llmConnections} 
            icon={<LLMIcon />} 
            color={theme.palette.info.main} 
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4} lg={2.4}>
          <StatCard 
            title="Users" 
            value={data.stats.users} 
            icon={<UserIcon />} 
            color={theme.palette.warning.main} 
          />
        </Grid>
      </Grid>

      {/* Activity charts and lists */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              System Activity
            </Typography>
            <Box 
              sx={{ 
                height: 300, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                bgcolor: 'background.default',
                borderRadius: 1
              }}
            >
              <Typography color="text.secondary">
                Activity chart will appear here
              </Typography>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              System Logs
            </Typography>
            <List>
              {data.systemLogs.map((log, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem alignItems="flex-start">
                    <ListItemText
                      primary={
                        <Box display="flex" justifyContent="space-between">
                          <Typography 
                            component="span" 
                            variant="body2"
                            sx={{ 
                              bgcolor: log.level === 'ERROR' ? 'error.light' : 
                                      log.level === 'WARNING' ? 'warning.light' : 'info.light',
                              color: log.level === 'ERROR' ? 'error.contrastText' : 
                                     log.level === 'WARNING' ? 'warning.contrastText' : 'info.contrastText',
                              px: 1,
                              py: 0.5,
                              borderRadius: 1,
                              fontSize: '0.75rem',
                              fontWeight: 'bold'
                            }}
                          >
                            {log.level}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {log.time}
                          </Typography>
                        </Box>
                      }
                      secondary={log.message}
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Agents
            </Typography>
            <List>
              {data.recentAgents.map((agent, index) => (
                <React.Fragment key={agent.id}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem>
                    <ListItemIcon>
                      <AgentIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={agent.name}
                      secondary={agent.type}
                    />
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        bgcolor: agent.status === 'Active' ? 'success.light' : 'text.disabled',
                        color: agent.status === 'Active' ? 'success.contrastText' : 'background.paper',
                        px: 1,
                        py: 0.5,
                        borderRadius: 1
                      }}
                    >
                      {agent.status}
                    </Typography>
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Tools
            </Typography>
            <List>
              {data.recentTools.map((tool, index) => (
                <React.Fragment key={tool.id}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem>
                    <ListItemIcon>
                      <ToolIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={tool.name}
                      secondary={tool.type}
                    />
                    <Typography 
                      variant="caption" 
                      sx={{ 
                        bgcolor: 
                          tool.usage === 'High' ? 'success.light' : 
                          tool.usage === 'Medium' ? 'warning.light' : 'text.disabled',
                        color: 
                          tool.usage === 'High' ? 'success.contrastText' : 
                          tool.usage === 'Medium' ? 'warning.contrastText' : 'background.paper',
                        px: 1,
                        py: 0.5,
                        borderRadius: 1
                      }}
                    >
                      {tool.usage}
                    </Typography>
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
