import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { PageHeader } from '../../components/common';
import { AgentDetail, Agent } from '../../components/agents';
import axios from 'axios';

const AgentDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch agent details
  const fetchAgentDetails = useCallback(async () => {
    if (!id) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`/api/agents/${id}`);
      // setAgent(response.data);
      
      // Using mock data for now
      const mockAgent: Agent = {
        id: parseInt(id),
        name: 'Customer Support Bot',
        description: 'Handles customer inquiries and support tickets',
        agent_type: 'MemoryEnhancedAgent',
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        creator_id: 1,
        team_id: null,
        configuration: {
          memory_size: 10,
          memory_type: 'episodic',
          prompt_template: 'You are a helpful customer support agent. {input} {memory}'
        }
      };
      
      setAgent(mockAgent);
    } catch (err: any) {
      console.error('Error fetching agent details:', err);
      setError('Failed to load agent details. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  // Load agent details on component mount
  useEffect(() => {
    fetchAgentDetails();
  }, [fetchAgentDetails]);

  // Handle delete agent
  const handleDeleteAgent = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/agents/${id}`);
      
      // Navigate back to agents list
      navigate('/agents');
    } catch (err: any) {
      console.error('Error deleting agent:', err);
      setError('Failed to delete agent. Please try again.');
    }
  };

  // Handle toggle active
  const handleToggleActive = async (id: number, isActive: boolean) => {
    try {
      // In a real app, this would be an API call
      // await axios.patch(`/api/agents/${id}`, { is_active: isActive });
      
      // Update agent state
      setAgent(prevAgent => {
        if (!prevAgent) return null;
        return { ...prevAgent, is_active: isActive };
      });
    } catch (err: any) {
      console.error('Error toggling agent active state:', err);
      setError('Failed to update agent. Please try again.');
    }
  };

  // Handle run agent
  const handleRunAgent = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.post(`/api/agents/${id}/run`);
      
      // Show success message
      alert('Agent execution started successfully!');
    } catch (err: any) {
      console.error('Error running agent:', err);
      setError('Failed to run agent. Please try again.');
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Agent Details"
        breadcrumbs={[
          { label: 'Agents', path: '/agents' },
          { label: agent?.name || 'Agent Details' }
        ]}
      />
      
      <AgentDetail
        agent={agent}
        isLoading={isLoading}
        error={error}
        onDelete={handleDeleteAgent}
        onToggleActive={handleToggleActive}
        onRun={handleRunAgent}
      />
    </Box>
  );
};

export default AgentDetailPage;
