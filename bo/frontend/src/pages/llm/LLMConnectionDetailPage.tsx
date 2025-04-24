import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Tabs,
  Tab,
  CircularProgress,
  Breadcrumbs
} from '@mui/material';
import { 
  ArrowBack as ArrowBackIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { Link, useParams } from 'react-router-dom';

const LLMConnectionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [tabValue, setTabValue] = React.useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      {/* Breadcrumbs */}
      <Breadcrumbs sx={{ mb: 2 }}>
        <Link to="/llm" style={{ textDecoration: 'none', color: 'inherit' }}>
          LLM Connections
        </Link>
        <Typography color="text.primary">Connection Details</Typography>
      </Breadcrumbs>

      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box display="flex" alignItems="center">
          <Button 
            component={Link} 
            to="/llm" 
            startIcon={<ArrowBackIcon />}
            sx={{ mr: 2 }}
          >
            Back
          </Button>
          <Typography variant="h4">Connection Details</Typography>
        </Box>
        <Box>
          <Button 
            variant="contained" 
            color="info" 
            startIcon={<RefreshIcon />}
            sx={{ mr: 1 }}
          >
            Test Connection
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
      </Box>
      
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
          <Tab label="Usage" />
          <Tab label="Models" />
        </Tabs>
      </Paper>
      
      {/* Tab Content */}
      <Paper sx={{ p: 3 }}>
        {tabValue === 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>Overview</Typography>
            <Typography variant="body1" color="text.secondary" align="center" py={8}>
              LLM connection overview will be implemented here.
            </Typography>
          </Box>
        )}
        {tabValue === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>Configuration</Typography>
            <Typography variant="body1" color="text.secondary" align="center" py={8}>
              LLM connection configuration will be implemented here.
            </Typography>
          </Box>
        )}
        {tabValue === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>Usage</Typography>
            <Typography variant="body1" color="text.secondary" align="center" py={8}>
              LLM connection usage statistics will be implemented here.
            </Typography>
          </Box>
        )}
        {tabValue === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>Models</Typography>
            <Typography variant="body1" color="text.secondary" align="center" py={8}>
              Available LLM models will be listed here.
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default LLMConnectionDetailPage;
