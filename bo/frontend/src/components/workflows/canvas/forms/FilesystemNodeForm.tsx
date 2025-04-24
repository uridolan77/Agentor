import React from 'react';
import { Grid, Typography } from '@mui/material';
import { FormField } from '../../../common';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData } from '../types/nodeTypes';

/**
 * Form component for configuring a Filesystem node.
 * This node enables filesystem operations in workflows.
 */
const FilesystemNodeForm: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    operation_type?: string;
    path_template?: string;
    content_template?: string;
    recursive?: boolean;
    create_dirs?: boolean;
    encoding?: string;
    error_if_exists?: boolean;
    error_if_not_exists?: boolean;
  }>({
    operation_type: 'read',
    path_template: '',
    content_template: '',
    recursive: false,
    create_dirs: true,
    encoding: 'utf-8',
    error_if_exists: false,
    error_if_not_exists: true
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        operation_type: node.config.operation_type || 'read',
        path_template: node.config.path_template || '',
        content_template: node.config.content_template || '',
        recursive: node.config.recursive || false,
        create_dirs: node.config.create_dirs !== false,
        encoding: node.config.encoding || 'utf-8',
        error_if_exists: node.config.error_if_exists || false,
        error_if_not_exists: node.config.error_if_not_exists !== false
      });
    } else {
      setFormValues({
        operation_type: 'read',
        path_template: '',
        content_template: '',
        recursive: false,
        create_dirs: true,
        encoding: 'utf-8',
        error_if_exists: false,
        error_if_not_exists: true
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
        operation_type: formValues.operation_type,
        path_template: formValues.path_template,
        content_template: formValues.content_template,
        recursive: formValues.recursive,
        create_dirs: formValues.create_dirs,
        encoding: formValues.encoding,
        error_if_exists: formValues.error_if_exists,
        error_if_not_exists: formValues.error_if_not_exists
      }
    };
    
    onSave(nodeWithConfig);
  };

  // Determine if content template should be shown
  const showContentTemplate = formValues.operation_type === 'write' || 
                             formValues.operation_type === 'append';

  return (
    <NodeFormBase
      open={open}
      node={node}
      onClose={onClose}
      onSave={handleSave}
      isLoading={isLoading}
    >
      <Typography variant="subtitle1" gutterBottom>
        Filesystem Configuration
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FormField
            type="select"
            name="operation_type"
            label="Operation Type"
            value={formValues.operation_type || 'read'}
            onChange={handleChange}
            options={[
              { value: 'read', label: 'Read File' },
              { value: 'write', label: 'Write File' },
              { value: 'append', label: 'Append to File' },
              { value: 'delete', label: 'Delete File' },
              { value: 'list', label: 'List Directory' },
              { value: 'exists', label: 'Check if Exists' },
              { value: 'copy', label: 'Copy File' },
              { value: 'move', label: 'Move File' }
            ]}
            helperText="Type of filesystem operation to perform"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="text"
            name="path_template"
            label="Path Template"
            value={formValues.path_template || ''}
            onChange={handleChange}
            helperText="File or directory path template (can include placeholders like {variable})"
          />
        </Grid>

        {showContentTemplate && (
          <Grid item xs={12}>
            <FormField
              type="textarea"
              name="content_template"
              label="Content Template"
              value={formValues.content_template || ''}
              onChange={handleChange}
              rows={4}
              helperText="Content template for write/append operations (can include placeholders)"
            />
          </Grid>
        )}

        <Grid item xs={12} md={6}>
          <FormField
            type="select"
            name="encoding"
            label="File Encoding"
            value={formValues.encoding || 'utf-8'}
            onChange={handleChange}
            options={[
              { value: 'utf-8', label: 'UTF-8' },
              { value: 'ascii', label: 'ASCII' },
              { value: 'latin1', label: 'Latin-1' },
              { value: 'binary', label: 'Binary' }
            ]}
            helperText="Encoding for text file operations"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="recursive"
            label="Recursive Operation"
            value={formValues.recursive || false}
            onChange={handleChange}
            helperText="Apply operation recursively (for directory operations)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="create_dirs"
            label="Create Directories"
            value={formValues.create_dirs !== false}
            onChange={handleChange}
            helperText="Create parent directories if they don't exist"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="error_if_exists"
            label="Error if Exists"
            value={formValues.error_if_exists || false}
            onChange={handleChange}
            helperText="Raise error if file already exists (for write operations)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="error_if_not_exists"
            label="Error if Not Exists"
            value={formValues.error_if_not_exists !== false}
            onChange={handleChange}
            helperText="Raise error if file doesn't exist (for read operations)"
          />
        </Grid>
      </Grid>
    </NodeFormBase>
  );
};

export default FilesystemNodeForm;
