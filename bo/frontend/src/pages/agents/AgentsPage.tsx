import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { PageHeader } from '../../components/common';
import { AgentList, Agent } from '../../components/agents';
import axios from 'axios';

const AgentsPage: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch agents
  const fetchAgents = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('/api/agents');
      // setAgents(response.data);
      
      // Using mock data for now
      const mockAgents: Agent[] = [
        {
          id: 1,
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
        },
        {
          id: 2,
          name: 'Data Processor',
          description: 'Processes and analyzes data from various sources',
          agent_type: 'ReactiveAgent',
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          creator_id: 1,
          team_id: null,
          configuration: {
            max_iterations: 5,
            prompt_template: 'You are a data processing agent. {input}'
          }
        },
        {
          id: 3,
          name: 'Content Moderator',
          description: 'Moderates user-generated content',
          agent_type: 'RuleBasedAgent',
          is_active: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          creator_id: 1,
          team_id: null,
          configuration: {
            rules: JSON.stringify([
              { condition: 'contains_profanity', action: 'flag' },
              { condition: 'contains_pii', action: 'redact' }
            ]),
            rule_evaluation: 'all_matching'
          }
        }
      ];
      
      setAgents(mockAgents);
    } catch (err: any) {
      console.error('Error fetching agents:', err);
      setError('Failed to load agents. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load agents on component mount
  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  // Handle delete agent
  const handleDeleteAgent = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/agents/${id}`);
      
      // For now, just remove from state
      setAgents(prevAgents => prevAgents.filter(agent => agent.id !== id));
    } catch (err: any) {
      console.error('Error deleting agent:', err);
      setError('Failed to delete agent. Please try again.');
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Agents"
        breadcrumbs={[{ label: 'Agents' }]}
      />
      
      <AgentList
        agents={agents}
        isLoading={isLoading}
        error={error}
        onRefresh={fetchAgents}
        onDelete={handleDeleteAgent}
      />
    </Box>
  );
};

export default AgentsPage;
