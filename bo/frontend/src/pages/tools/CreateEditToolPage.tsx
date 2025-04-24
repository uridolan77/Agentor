import React, { useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import { PageHeader } from '../../components/common';
import { ToolForm, ToolFormValues, Tool } from '../../components/tools';
import axios from 'axios';

const CreateEditToolPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isEditMode = !!id;
  const [tool, setTool] = useState<Tool | null>(null);
  const [isLoading, setIsLoading] = useState(isEditMode);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Fetch tool details for edit mode
  const fetchToolDetails = useCallback(async () => {
    if (!isEditMode) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`/api/tools/${id}`);
      // setTool(response.data);
      
      // Using mock data for now
      const mockTool: Tool = {
        id: parseInt(id!),
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
  }, [id, isEditMode]);

  // Load tool details on component mount for edit mode
  useEffect(() => {
    fetchToolDetails();
  }, [fetchToolDetails]);

  // Handle form submission
  const handleSubmit = async (values: ToolFormValues) => {
    setIsSubmitting(true);
    setError(null);
    
    try {
      if (isEditMode) {
        // In a real app, this would be an API call
        // await axios.put(`/api/tools/${id}`, values);
        console.log('Updating tool:', values);
      } else {
        // In a real app, this would be an API call
        // const response = await axios.post('/api/tools', values);
        console.log('Creating tool:', values);
      }
      
      // Navigate back to tools list
      navigate('/tools');
    } catch (err: any) {
      console.error('Error saving tool:', err);
      setError('Failed to save tool. Please try again.');
      setIsSubmitting(false);
    }
  };

  // Prepare initial form values
  const getInitialValues = (): Partial<ToolFormValues> => {
    if (!isEditMode) {
      return {
        name: '',
        description: '',
        tool_type: 'BaseTool',
        is_active: true,
        configuration: {}
      };
    }
    
    if (!tool) return {};
    
    return {
      name: tool.name,
      description: tool.description || '',
      tool_type: tool.tool_type,
      is_active: tool.is_active,
      configuration: typeof tool.configuration === 'string'
        ? JSON.parse(tool.configuration)
        : tool.configuration
    };
  };

  return (
    <Box>
      <PageHeader 
        title={isEditMode ? 'Edit Tool' : 'Create Tool'}
        breadcrumbs={[
          { label: 'Tools', path: '/tools' },
          { label: isEditMode ? 'Edit Tool' : 'Create Tool' }
        ]}
      />
      
      <ToolForm
        initialValues={getInitialValues()}
        onSubmit={handleSubmit}
        isLoading={isLoading || isSubmitting}
        error={error}
        mode={isEditMode ? 'edit' : 'create'}
      />
    </Box>
  );
};

export default CreateEditToolPage;
