import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { PageHeader } from '../../components/common';
import { AgentForm, AgentFormValues, Agent } from '../../components/agents';
import axios from 'axios';

const CreateEditAgentPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isEditMode = !!id;
  const [agent, setAgent] = useState<Agent | null>(null);
  const [isLoading, setIsLoading] = useState(isEditMode);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Fetch agent details for edit mode
  const fetchAgentDetails = useCallback(async () => {
    if (!isEditMode) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`/api/agents/${id}`);
      // setAgent(response.data);
      
      // Using mock data for now
      const mockAgent: Agent = {
        id: parseInt(id!),
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
  }, [id, isEditMode]);

  // Load agent details on component mount for edit mode
  useEffect(() => {
    fetchAgentDetails();
  }, [fetchAgentDetails]);

  // Handle form submission
  const handleSubmit = async (values: AgentFormValues) => {
    setIsSubmitting(true);
    setError(null);
    
    try {
      if (isEditMode) {
        // In a real app, this would be an API call
        // await axios.put(`/api/agents/${id}`, values);
        console.log('Updating agent:', values);
      } else {
        // In a real app, this would be an API call
        // const response = await axios.post('/api/agents', values);
        console.log('Creating agent:', values);
      }
      
      // Navigate back to agents list
      navigate('/agents');
    } catch (err: any) {
      console.error('Error saving agent:', err);
      setError('Failed to save agent. Please try again.');
      setIsSubmitting(false);
    }
  };

  // Prepare initial form values
  const getInitialValues = (): Partial<AgentFormValues> => {
    if (!isEditMode) {
      return {
        name: '',
        description: '',
        agent_type: 'ReactiveAgent',
        is_active: true,
        configuration: {}
      };
    }
    
    if (!agent) return {};
    
    return {
      name: agent.name,
      description: agent.description || '',
      agent_type: agent.agent_type,
      is_active: agent.is_active,
      configuration: typeof agent.configuration === 'string'
        ? JSON.parse(agent.configuration)
        : agent.configuration
    };
  };

  return (
    <Box>
      <PageHeader 
        title={isEditMode ? 'Edit Agent' : 'Create Agent'}
        breadcrumbs={[
          { label: 'Agents', path: '/agents' },
          { label: isEditMode ? 'Edit Agent' : 'Create Agent' }
        ]}
      />
      
      <AgentForm
        initialValues={getInitialValues()}
        onSubmit={handleSubmit}
        isLoading={isLoading || isSubmitting}
        error={error}
        mode={isEditMode ? 'edit' : 'create'}
      />
    </Box>
  );
};

export default CreateEditAgentPage;
