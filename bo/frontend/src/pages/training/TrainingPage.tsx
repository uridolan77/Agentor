import React, { useState } from 'react';
import { Box, Typography, Paper, Tabs, Tab, Button } from '@mui/material';
import { PageHeader } from '../../components/common';
import { TrainingDashboard } from '../../components/training';
import { useParams, useNavigate } from 'react-router-dom';

/**
 * Training page component
 * Displays the training dashboard for agent model training
 */
const TrainingPage: React.FC = () => {
  const { agentId } = useParams<{ agentId?: string }>();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle training complete
  const handleTrainingComplete = (sessionId: string) => {
    // If we have an agent ID, navigate back to the agent detail page
    if (agentId) {
      navigate(`/agents/${agentId}`);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <PageHeader 
        title={agentId ? "Agent Training" : "Model Training"} 
        subtitle={agentId 
          ? "Train and fine-tune models for this agent" 
          : "Manage training sessions and models"
        }
      />

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Training Dashboard" />
          <Tab label="Models" />
          <Tab label="Datasets" />
        </Tabs>

        {/* Training Dashboard Tab */}
        {tabValue === 0 && (
          <Box sx={{ p: 2 }}>
            <TrainingDashboard 
              agentId={agentId} 
              onTrainingComplete={handleTrainingComplete}
            />
          </Box>
        )}

        {/* Models Tab */}
        {tabValue === 1 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Available Models
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
              This section will display available models and allow you to manage them.
            </Typography>
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => setTabValue(0)}
            >
              Train New Model
            </Button>
          </Box>
        )}

        {/* Datasets Tab */}
        {tabValue === 2 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Available Datasets
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
              This section will display available datasets and allow you to upload new ones.
            </Typography>
            <Button 
              variant="contained" 
              color="primary"
            >
              Upload New Dataset
            </Button>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default TrainingPage;
