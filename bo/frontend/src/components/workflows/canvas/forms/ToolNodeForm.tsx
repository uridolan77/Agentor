import React from 'react';
import {
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Switch,
  FormControlLabel,
  Divider,
  Typography
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';

interface ToolNodeFormProps extends Omit<NodeFormProps, 'children'> {
  availableTools?: { id: string; name: string; category: string }[];
}

/**
 * Form for editing Tool nodes
 */
const ToolNodeForm: React.FC<ToolNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading,
  availableTools = []
}) => {
  const [formValues, setFormValues] = React.useState<{
    toolId?: string;
    parameters?: Record<string, any>;
    retryOnFailure?: boolean;
    timeout?: number;
  }>({
    toolId: '',
    parameters: {},
    retryOnFailure: false,
    timeout: 30
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        toolId: node.config.toolId || '',
        parameters: node.config.parameters || {},
        retryOnFailure: node.config.retryOnFailure || false,
        timeout: node.config.timeout || 30
      });
    } else {
      setFormValues({
        toolId: '',
        parameters: {},
        retryOnFailure: false,
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

  // Handle parameter change
  const handleParameterChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      parameters: {
        ...(prev.parameters || {}),
        [name]: value
      }
    }));
  };

  // Handle save
  const handleSave = (updatedNode: NodeData) => {
    const nodeWithConfig: NodeData = {
      ...updatedNode,
      config: {
        ...updatedNode.config,
        toolId: formValues.toolId,
        parameters: formValues.parameters,
        retryOnFailure: formValues.retryOnFailure,
        timeout: formValues.timeout
      }
    };
    
    onSave(nodeWithConfig);
  };

  // Group tools by category
  const toolsByCategory = React.useMemo(() => {
    const grouped: Record<string, { id: string; name: string }[]> = {};
    
    availableTools.forEach(tool => {
      if (!grouped[tool.category]) {
        grouped[tool.category] = [];
      }
      grouped[tool.category].push({ id: tool.id, name: tool.name });
    });
    
    return grouped;
  }, [availableTools]);

  // Get selected tool
  const selectedTool = React.useMemo(() => {
    return availableTools.find(tool => tool.id === formValues.toolId);
  }, [availableTools, formValues.toolId]);

  return (
    <NodeFormBase
      open={open}
      node={node}
      onClose={onClose}
      onSave={handleSave}
      isLoading={isLoading}
    >
      {/* Tool Selection */}
      <Grid item xs={12}>
        <FormControl fullWidth>
          <InputLabel id="tool-select-label">Tool</InputLabel>
          <Select
            labelId="tool-select-label"
            id="tool-select"
            value={formValues.toolId || ''}
            label="Tool"
            onChange={(e) => handleChange('toolId', e.target.value)}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            
            {Object.entries(toolsByCategory).map(([category, tools]) => [
              <MenuItem key={`category-${category}`} disabled divider>
                {category}
              </MenuItem>,
              ...tools.map(tool => (
                <MenuItem key={tool.id} value={tool.id}>
                  {tool.name}
                </MenuItem>
              ))
            ])}
          </Select>
          <FormHelperText>Select the tool to use in this node</FormHelperText>
        </FormControl>
      </Grid>
      
      {/* Tool Settings */}
      <Grid item xs={12}>
        <Typography variant="subtitle2" gutterBottom>
          Tool Settings
        </Typography>
        <Divider sx={{ mb: 2 }} />
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Timeout (seconds)"
          name="timeout"
          type="number"
          inputProps={{ min: 1, step: 1 }}
          value={formValues.timeout || 30}
          onChange={(e) => handleChange('timeout', parseInt(e.target.value))}
          helperText="Maximum execution time in seconds"
        />
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <FormControlLabel
          control={
            <Switch
              checked={formValues.retryOnFailure || false}
              onChange={(e) => handleChange('retryOnFailure', e.target.checked)}
              name="retryOnFailure"
            />
          }
          label="Retry on failure"
        />
      </Grid>
      
      {/* Tool Parameters */}
      {selectedTool && (
        <>
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Tool Parameters
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          {/* In a real app, we would dynamically generate parameter fields based on the tool's schema */}
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Input"
              name="input"
              multiline
              rows={3}
              value={formValues.parameters?.input || ''}
              onChange={(e) => handleParameterChange('input', e.target.value)}
              helperText="Input data for the tool"
            />
          </Grid>
          
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={formValues.parameters?.async || false}
                  onChange={(e) => handleParameterChange('async', e.target.checked)}
                  name="async"
                />
              }
              label="Run asynchronously"
            />
          </Grid>
        </>
      )}
    </NodeFormBase>
  );
};

export default ToolNodeForm;
