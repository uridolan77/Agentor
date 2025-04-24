import React from 'react';
import { Grid, Typography } from '@mui/material';
import { FormField } from '../../../common';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData } from '../types/nodeTypes';

/**
 * Form component for configuring an API node.
 * This node enables API operations in workflows.
 */
const APINodeForm: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    method?: string;
    url_template?: string;
    headers_template?: string;
    body_template?: string;
    auth_type?: string;
    auth_username?: string;
    auth_token?: string;
    timeout?: number;
    retry_count?: number;
    follow_redirects?: boolean;
  }>({
    method: 'GET',
    url_template: '',
    headers_template: '{}',
    body_template: '',
    auth_type: 'none',
    auth_username: '',
    auth_token: '',
    timeout: 30,
    retry_count: 3,
    follow_redirects: true
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        method: node.config.method || 'GET',
        url_template: node.config.url_template || '',
        headers_template: node.config.headers_template || '{}',
        body_template: node.config.body_template || '',
        auth_type: node.config.auth_type || 'none',
        auth_username: node.config.auth_username || '',
        auth_token: node.config.auth_token || '',
        timeout: node.config.timeout || 30,
        retry_count: node.config.retry_count || 3,
        follow_redirects: node.config.follow_redirects !== false
      });
    } else {
      setFormValues({
        method: 'GET',
        url_template: '',
        headers_template: '{}',
        body_template: '',
        auth_type: 'none',
        auth_username: '',
        auth_token: '',
        timeout: 30,
        retry_count: 3,
        follow_redirects: true
      });
    }
  }, [node]);

  // Handle form change
  const handleChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle save
  const handleSave = (updatedNode: NodeData) => {
    const nodeWithConfig: NodeData = {
      ...updatedNode,
      config: {
        ...updatedNode.config,
        method: formValues.method,
        url_template: formValues.url_template,
        headers_template: formValues.headers_template,
        body_template: formValues.body_template,
        auth_type: formValues.auth_type,
        auth_username: formValues.auth_username,
        auth_token: formValues.auth_token,
        timeout: formValues.timeout,
        retry_count: formValues.retry_count,
        follow_redirects: formValues.follow_redirects
      }
    };
    
    onSave(nodeWithConfig);
  };

  // Determine if body template should be shown
  const showBodyTemplate = formValues.method === 'POST' || 
                          formValues.method === 'PUT' || 
                          formValues.method === 'PATCH';

  // Determine if auth fields should be shown
  const showAuthFields = formValues.auth_type !== 'none';

  return (
    <NodeFormBase
      open={open}
      node={node}
      onClose={onClose}
      onSave={handleSave}
      isLoading={isLoading}
    >
      <Typography variant="subtitle1" gutterBottom>
        API Configuration
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <FormField
            type="select"
            name="method"
            label="HTTP Method"
            value={formValues.method || 'GET'}
            onChange={handleChange}
            options={[
              { value: 'GET', label: 'GET' },
              { value: 'POST', label: 'POST' },
              { value: 'PUT', label: 'PUT' },
              { value: 'DELETE', label: 'DELETE' },
              { value: 'PATCH', label: 'PATCH' },
              { value: 'HEAD', label: 'HEAD' },
              { value: 'OPTIONS', label: 'OPTIONS' }
            ]}
            helperText="HTTP method for the API request"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="select"
            name="auth_type"
            label="Authentication Type"
            value={formValues.auth_type || 'none'}
            onChange={handleChange}
            options={[
              { value: 'none', label: 'None' },
              { value: 'basic', label: 'Basic Auth' },
              { value: 'bearer', label: 'Bearer Token' },
              { value: 'api_key', label: 'API Key' },
              { value: 'oauth2', label: 'OAuth 2.0' }
            ]}
            helperText="Authentication method for the API"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="text"
            name="url_template"
            label="URL Template"
            value={formValues.url_template || ''}
            onChange={handleChange}
            helperText="API URL template (can include placeholders like {variable})"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="textarea"
            name="headers_template"
            label="Headers Template (JSON)"
            value={formValues.headers_template || '{}'}
            onChange={handleChange}
            rows={3}
            helperText="JSON template for request headers (can include placeholders)"
          />
        </Grid>

        {showBodyTemplate && (
          <Grid item xs={12}>
            <FormField
              type="textarea"
              name="body_template"
              label="Body Template"
              value={formValues.body_template || ''}
              onChange={handleChange}
              rows={4}
              helperText="Request body template (can include placeholders)"
            />
          </Grid>
        )}

        {showAuthFields && (
          <>
            <Grid item xs={12} md={6}>
              <FormField
                type="text"
                name="auth_username"
                label={formValues.auth_type === 'api_key' ? 'API Key Name' : 'Username'}
                value={formValues.auth_username || ''}
                onChange={handleChange}
                helperText={formValues.auth_type === 'api_key' ? 'Name of the API key header/parameter' : 'Username for authentication'}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormField
                type="password"
                name="auth_token"
                label={formValues.auth_type === 'api_key' ? 'API Key Value' : 
                       formValues.auth_type === 'bearer' ? 'Bearer Token' : 
                       formValues.auth_type === 'oauth2' ? 'OAuth Token' : 'Password'}
                value={formValues.auth_token || ''}
                onChange={handleChange}
                helperText="Authentication token/password (will be encrypted)"
              />
            </Grid>
          </>
        )}

        <Grid item xs={12} md={4}>
          <FormField
            type="number"
            name="timeout"
            label="Timeout (seconds)"
            value={formValues.timeout || 30}
            onChange={handleChange}
            min={1}
            max={300}
            helperText="Request timeout in seconds"
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <FormField
            type="number"
            name="retry_count"
            label="Retry Count"
            value={formValues.retry_count || 3}
            onChange={handleChange}
            min={0}
            max={10}
            helperText="Number of retry attempts on failure"
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <FormField
            type="switch"
            name="follow_redirects"
            label="Follow Redirects"
            value={formValues.follow_redirects !== false}
            onChange={handleChange}
            helperText="Automatically follow HTTP redirects"
          />
        </Grid>
      </Grid>
    </NodeFormBase>
  );
};

export default APINodeForm;
