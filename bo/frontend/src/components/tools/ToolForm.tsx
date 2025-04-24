import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Button,
  Grid,
  Typography,
  Divider,
  Alert,
  CircularProgress
} from '@mui/material';
import { FormField, FieldOption } from '../common';
import { Tool } from './ToolList';

// Define tool types
const TOOL_TYPES: FieldOption[] = [
  { value: 'BaseTool', label: 'Base Tool' },
  { value: 'EnhancedTool', label: 'Enhanced Tool' },
  { value: 'ComposableTool', label: 'Composable Tool' },
  { value: 'APITool', label: 'API Tool' }
];

// Define form values type
export interface ToolFormValues {
  name: string;
  description: string;
  tool_type: string;
  is_active: boolean;
  configuration: Record<string, any>;
}

// Define form errors type
interface FormErrors {
  name?: string;
  description?: string;
  tool_type?: string;
  configuration?: string;
}

interface ToolFormProps {
  initialValues?: Partial<ToolFormValues>;
  onSubmit: (values: ToolFormValues) => Promise<void>;
  isLoading?: boolean;
  error?: string | null;
  mode: 'create' | 'edit';
}

/**
 * ToolForm component for creating and editing tools.
 * Includes form validation and configuration based on tool type.
 */
const ToolForm: React.FC<ToolFormProps> = ({
  initialValues,
  onSubmit,
  isLoading = false,
  error = null,
  mode
}) => {
  const navigate = useNavigate();
  const [values, setValues] = useState<ToolFormValues>({
    name: '',
    description: '',
    tool_type: 'BaseTool',
    is_active: true,
    configuration: {}
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // Initialize form values from initialValues
  useEffect(() => {
    if (initialValues) {
      setValues(prevValues => ({
        ...prevValues,
        ...initialValues,
        configuration: initialValues.configuration || {}
      }));
    }
  }, [initialValues]);

  // Handle field change
  const handleChange = (name: string, value: any) => {
    setValues(prevValues => ({
      ...prevValues,
      [name]: value
    }));

    // Mark field as touched
    if (!touched[name]) {
      setTouched(prevTouched => ({
        ...prevTouched,
        [name]: true
      }));
    }

    // Clear error when field is changed
    if (errors[name as keyof FormErrors]) {
      setErrors(prevErrors => ({
        ...prevErrors,
        [name]: undefined
      }));
    }

    // Reset configuration when tool type changes
    if (name === 'tool_type') {
      setValues(prevValues => ({
        ...prevValues,
        configuration: {}
      }));
    }
  };

  // Handle configuration field change
  const handleConfigChange = (name: string, value: any) => {
    setValues(prevValues => ({
      ...prevValues,
      configuration: {
        ...prevValues.configuration,
        [name]: value
      }
    }));
  };

  // Validate form
  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // Validate name
    if (!values.name.trim()) {
      newErrors.name = 'Name is required';
    } else if (values.name.length > 100) {
      newErrors.name = 'Name must be less than 100 characters';
    }

    // Validate description (optional)
    if (values.description && values.description.length > 500) {
      newErrors.description = 'Description must be less than 500 characters';
    }

    // Validate tool type
    if (!values.tool_type) {
      newErrors.tool_type = 'Tool type is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Mark all fields as touched
    const allTouched = Object.keys(values).reduce((acc, key) => {
      acc[key] = true;
      return acc;
    }, {} as Record<string, boolean>);
    setTouched(allTouched);

    // Validate form
    if (!validateForm()) {
      return;
    }

    // Submit form
    try {
      await onSubmit(values);
      navigate('/tools');
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  // Handle cancel
  const handleCancel = () => {
    navigate('/tools');
  };

  // Render configuration fields based on tool type
  const renderConfigurationFields = () => {
    switch (values.tool_type) {
      case 'BaseTool':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="command"
                label="Command"
                value={values.configuration.command || ''}
                onChange={handleConfigChange}
                rows={3}
                helperText="Command to execute when the tool is invoked"
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="checkbox"
                name="requires_input"
                label="Requires Input"
                value={values.configuration.requires_input || false}
                onChange={handleConfigChange}
              />
            </Grid>
          </>
        );
        
      case 'EnhancedTool':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="command"
                label="Command"
                value={values.configuration.command || ''}
                onChange={handleConfigChange}
                rows={3}
                helperText="Command to execute when the tool is invoked"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="timeout"
                label="Timeout (seconds)"
                value={values.configuration.timeout || 30}
                onChange={handleConfigChange}
                min={1}
                max={300}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="error_handling"
                label="Error Handling"
                value={values.configuration.error_handling || 'retry'}
                onChange={handleConfigChange}
                options={[
                  { value: 'retry', label: 'Retry' },
                  { value: 'fail', label: 'Fail' },
                  { value: 'ignore', label: 'Ignore' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="checkbox"
                name="requires_input"
                label="Requires Input"
                value={values.configuration.requires_input || false}
                onChange={handleConfigChange}
              />
            </Grid>
          </>
        );
        
      case 'ComposableTool':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="tools"
                label="Tools"
                value={values.configuration.tools || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="JSON array of tool IDs to compose"
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="select"
                name="execution_mode"
                label="Execution Mode"
                value={values.configuration.execution_mode || 'sequential'}
                onChange={handleConfigChange}
                options={[
                  { value: 'sequential', label: 'Sequential' },
                  { value: 'parallel', label: 'Parallel' },
                  { value: 'conditional', label: 'Conditional' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="execution_rules"
                label="Execution Rules"
                value={values.configuration.execution_rules || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="JSON configuration for execution rules (for conditional mode)"
              />
            </Grid>
          </>
        );
        
      case 'APITool':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="text"
                name="api_url"
                label="API URL"
                value={values.configuration.api_url || ''}
                onChange={handleConfigChange}
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="method"
                label="HTTP Method"
                value={values.configuration.method || 'GET'}
                onChange={handleConfigChange}
                options={[
                  { value: 'GET', label: 'GET' },
                  { value: 'POST', label: 'POST' },
                  { value: 'PUT', label: 'PUT' },
                  { value: 'DELETE', label: 'DELETE' },
                  { value: 'PATCH', label: 'PATCH' }
                ]}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="auth_type"
                label="Authentication Type"
                value={values.configuration.auth_type || 'none'}
                onChange={handleConfigChange}
                options={[
                  { value: 'none', label: 'None' },
                  { value: 'basic', label: 'Basic Auth' },
                  { value: 'bearer', label: 'Bearer Token' },
                  { value: 'api_key', label: 'API Key' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="headers"
                label="Headers"
                value={values.configuration.headers || ''}
                onChange={handleConfigChange}
                rows={3}
                helperText="JSON object of HTTP headers"
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="body_template"
                label="Body Template"
                value={values.configuration.body_template || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="Template for request body. Use {input} as a placeholder for tool input."
              />
            </Grid>
          </>
        );
        
      default:
        return null;
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <form onSubmit={handleSubmit}>
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Basic Information
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormField
              type="text"
              name="name"
              label="Tool Name"
              value={values.name}
              onChange={handleChange}
              error={touched.name ? errors.name : undefined}
              required
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="tool_type"
              label="Tool Type"
              value={values.tool_type}
              onChange={handleChange}
              options={TOOL_TYPES}
              error={touched.tool_type ? errors.tool_type : undefined}
              required
            />
          </Grid>

          <Grid item xs={12}>
            <FormField
              type="textarea"
              name="description"
              label="Description"
              value={values.description}
              onChange={handleChange}
              error={touched.description ? errors.description : undefined}
              rows={3}
            />
          </Grid>

          <Grid item xs={12}>
            <FormField
              type="switch"
              name="is_active"
              label="Active"
              value={values.is_active}
              onChange={handleChange}
            />
          </Grid>

          {/* Configuration */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Configuration
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>

          {renderConfigurationFields()}

          {/* Form Actions */}
          <Grid item xs={12} sx={{ mt: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
              <Button
                variant="outlined"
                onClick={handleCancel}
                disabled={isLoading}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={isLoading}
                startIcon={isLoading ? <CircularProgress size={20} /> : null}
              >
                {mode === 'create' ? 'Create Tool' : 'Update Tool'}
              </Button>
            </Box>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default ToolForm;
