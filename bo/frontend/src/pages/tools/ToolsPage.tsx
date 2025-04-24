import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { PageHeader } from '../../components/common';
import { ToolList, Tool } from '../../components/tools';
import axios from 'axios';

const ToolsPage: React.FC = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch tools
  const fetchTools = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('/api/tools');
      // setTools(response.data);
      
      // Using mock data for now
      const mockTools: Tool[] = [
        {
          id: 1,
          name: 'Web Scraper',
          description: 'Extracts data from websites',
          tool_type: 'EnhancedTool',
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          creator_id: 1,
          team_id: null,
          configuration: {
            command: 'python scraper.py {url}',
            timeout: 60,
            error_handling: 'retry',
            requires_input: true
          }
        },
        {
          id: 2,
          name: 'Text Summarizer',
          description: 'Summarizes long text documents',
          tool_type: 'APITool',
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          creator_id: 1,
          team_id: null,
          configuration: {
            api_url: 'https://api.summarizer.example/v1/summarize',
            method: 'POST',
            auth_type: 'bearer',
            headers: JSON.stringify({
              'Content-Type': 'application/json'
            }),
            body_template: '{"text": "{input}", "max_length": 100}'
          }
        },
        {
          id: 3,
          name: 'Data Processor',
          description: 'Processes and transforms data files',
          tool_type: 'BaseTool',
          is_active: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          creator_id: 1,
          team_id: null,
          configuration: {
            command: 'node process-data.js {input}',
            requires_input: true
          }
        }
      ];
      
      setTools(mockTools);
    } catch (err: any) {
      console.error('Error fetching tools:', err);
      setError('Failed to load tools. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load tools on component mount
  useEffect(() => {
    fetchTools();
  }, [fetchTools]);

  // Handle delete tool
  const handleDeleteTool = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/tools/${id}`);
      
      // For now, just remove from state
      setTools(prevTools => prevTools.filter(tool => tool.id !== id));
    } catch (err: any) {
      console.error('Error deleting tool:', err);
      setError('Failed to delete tool. Please try again.');
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Tools"
        breadcrumbs={[{ label: 'Tools' }]}
      />
      
      <ToolList
        tools={tools}
        isLoading={isLoading}
        error={error}
        onRefresh={fetchTools}
        onDelete={handleDeleteTool}
      />
    </Box>
  );
};

export default ToolsPage;
