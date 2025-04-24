import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  CircularProgress,
  Grid
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { Link } from 'react-router-dom';

const LLMConnectionsPage: React.FC = () => {
  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4">LLM Connections</Typography>
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<AddIcon />}
          component={Link}
          to="/llm/new"
        >
          Add Connection
        </Button>
      </Box>
      
      <Paper sx={{ p: 3 }}>
        <Typography variant="body1" color="text.secondary" align="center" py={8}>
          LLM connection management interface will be implemented here.
        </Typography>
      </Paper>
    </Box>
  );
};

export default LLMConnectionsPage;
