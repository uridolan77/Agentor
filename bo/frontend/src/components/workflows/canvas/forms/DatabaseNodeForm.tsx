import React from 'react';
import { Grid, Typography } from '@mui/material';
import { FormField } from '../../../common';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData } from '../types/nodeTypes';

/**
 * Form component for configuring a Database node.
 * This node enables database operations in workflows.
 */
const DatabaseNodeForm: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    connection_string?: string;
    database_type?: string;
    query_type?: string;
    query_template?: string;
    use_parameters?: boolean;
    max_results?: number;
    timeout?: number;
  }>({
    connection_string: '',
    database_type: 'mysql',
    query_type: 'select',
    query_template: '',
    use_parameters: true,
    max_results: 100,
    timeout: 30
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        connection_string: node.config.connection_string || '',
        database_type: node.config.database_type || 'mysql',
        query_type: node.config.query_type || 'select',
        query_template: node.config.query_template || '',
        use_parameters: node.config.use_parameters !== false,
        max_results: node.config.max_results || 100,
        timeout: node.config.timeout || 30
      });
    } else {
      setFormValues({
        connection_string: '',
        database_type: 'mysql',
        query_type: 'select',
        query_template: '',
        use_parameters: true,
        max_results: 100,
        timeout: 30
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
        connection_string: formValues.connection_string,
        database_type: formValues.database_type,
        query_type: formValues.query_type,
        query_template: formValues.query_template,
        use_parameters: formValues.use_parameters,
        max_results: formValues.max_results,
        timeout: formValues.timeout
      }
    };
    
    onSave(nodeWithConfig);
  };

  return (
    <NodeFormBase
      open={open}
      node={node}
      onClose={onClose}
      onSave={handleSave}
      isLoading={isLoading}
    >
      <Typography variant="subtitle1" gutterBottom>
        Database Configuration
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FormField
            type="select"
            name="database_type"
            label="Database Type"
            value={formValues.database_type || 'mysql'}
            onChange={handleChange}
            options={[
              { value: 'mysql', label: 'MySQL' },
              { value: 'postgresql', label: 'PostgreSQL' },
              { value: 'sqlite', label: 'SQLite' },
              { value: 'mongodb', label: 'MongoDB' },
              { value: 'oracle', label: 'Oracle' },
              { value: 'sqlserver', label: 'SQL Server' }
            ]}
            helperText="Type of database to connect to"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="text"
            name="connection_string"
            label="Connection String"
            value={formValues.connection_string || ''}
            onChange={handleChange}
            helperText="Database connection string (credentials will be encrypted)"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="select"
            name="query_type"
            label="Query Type"
            value={formValues.query_type || 'select'}
            onChange={handleChange}
            options={[
              { value: 'select', label: 'SELECT' },
              { value: 'insert', label: 'INSERT' },
              { value: 'update', label: 'UPDATE' },
              { value: 'delete', label: 'DELETE' },
              { value: 'procedure', label: 'Stored Procedure' },
              { value: 'custom', label: 'Custom Query' }
            ]}
            helperText="Type of database operation to perform"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="textarea"
            name="query_template"
            label="Query Template"
            value={formValues.query_template || ''}
            onChange={handleChange}
            rows={4}
            helperText="SQL query template with parameter placeholders (e.g., SELECT * FROM users WHERE id = :id)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="max_results"
            label="Maximum Results"
            value={formValues.max_results || 100}
            onChange={handleChange}
            min={1}
            max={10000}
            helperText="Maximum number of results to return"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="timeout"
            label="Timeout (seconds)"
            value={formValues.timeout || 30}
            onChange={handleChange}
            min={1}
            max={300}
            helperText="Query timeout in seconds"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="switch"
            name="use_parameters"
            label="Use Parameterized Queries"
            value={formValues.use_parameters !== false}
            onChange={handleChange}
            helperText="Use parameterized queries for better security (recommended)"
          />
        </Grid>
      </Grid>
    </NodeFormBase>
  );
};

export default DatabaseNodeForm;
