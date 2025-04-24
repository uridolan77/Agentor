import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { PageHeader } from '../../components/common';
import { ToolDetail, Tool } from '../../components/tools';
import axios from 'axios';

const ToolDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [tool, setTool] = useState<Tool | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch tool details
  const fetchToolDetails = useCallback(async () => {
    if (!id) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`/api/tools/${id}`);
      // setTool(response.data);
      
      // Using mock data for now
      const mockTool: Tool = {
        id: parseInt(id),
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
      };
      
      setTool(mockTool);
    } catch (err: any) {
      console.error('Error fetching tool details:', err);
      setError('Failed to load tool details. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  // Load tool details on component mount
  useEffect(() => {
    fetchToolDetails();
  }, [fetchToolDetails]);

  // Handle delete tool
  const handleDeleteTool = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/tools/${id}`);
      
      // Navigate back to tools list
      navigate('/tools');
    } catch (err: any) {
      console.error('Error deleting tool:', err);
      setError('Failed to delete tool. Please try again.');
    }
  };

  // Handle toggle active
  const handleToggleActive = async (id: number, isActive: boolean) => {
    try {
      // In a real app, this would be an API call
      // await axios.patch(`/api/tools/${id}`, { is_active: isActive });
      
      // Update tool state
      setTool(prevTool => {
        if (!prevTool) return null;
        return { ...prevTool, is_active: isActive };
      });
    } catch (err: any) {
      console.error('Error toggling tool active state:', err);
      setError('Failed to update tool. Please try again.');
    }
  };

  // Handle run tool
  const handleRunTool = async (id: number) => {
    try {
      // In a real app, this would be an API call
      // await axios.post(`/api/tools/${id}/run`);
      
      // Show success message
      alert('Tool execution started successfully!');
    } catch (err: any) {
      console.error('Error running tool:', err);
      setError('Failed to run tool. Please try again.');
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Tool Details"
        breadcrumbs={[
          { label: 'Tools', path: '/tools' },
          { label: tool?.name || 'Tool Details' }
        ]}
      />
      
      <ToolDetail
        tool={tool}
        isLoading={isLoading}
        error={error}
        onDelete={handleDeleteTool}
        onToggleActive={handleToggleActive}
        onRun={handleRunTool}
      />
    </Box>
  );
};

export default ToolDetailPage;
