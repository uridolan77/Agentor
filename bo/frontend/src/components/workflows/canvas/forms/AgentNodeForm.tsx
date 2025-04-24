import React from 'react';
import {
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Autocomplete,
  Chip
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';

interface AgentNodeFormProps extends Omit<NodeFormProps, 'children'> {
  availableAgents?: { id: string; name: string }[];
}

/**
 * Form for editing Agent nodes
 */
const AgentNodeForm: React.FC<AgentNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading,
  availableAgents = []
}) => {
  const [formValues, setFormValues] = React.useState<{
    agentId?: string;
    parameters?: Record<string, string>;
    capabilities?: string[];
  }>({
    agentId: '',
    parameters: {},
    capabilities: []
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        agentId: node.config.agentId || '',
        parameters: node.config.parameters || {},
        capabilities: node.config.capabilities || []
      });
    } else {
      setFormValues({
        agentId: '',
        parameters: {},
        capabilities: []
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
  const handleParameterChange = (name: string, value: string) => {
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
        agentId: formValues.agentId,
        parameters: formValues.parameters,
        capabilities: formValues.capabilities
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
      {/* Agent Selection */}
      <Grid item xs={12}>
        <FormControl fullWidth>
          <InputLabel id="agent-select-label">Agent</InputLabel>
          <Select
            labelId="agent-select-label"
            id="agent-select"
            value={formValues.agentId || ''}
            label="Agent"
            onChange={(e) => handleChange('agentId', e.target.value)}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            {availableAgents.map(agent => (
              <MenuItem key={agent.id} value={agent.id}>
                {agent.name}
              </MenuItem>
            ))}
          </Select>
          <FormHelperText>Select the agent to use in this node</FormHelperText>
        </FormControl>
      </Grid>
      
      {/* Agent Capabilities */}
      <Grid item xs={12}>
        <Autocomplete
          multiple
          id="capabilities"
          options={[
            'text-generation',
            'image-generation',
            'code-generation',
            'data-analysis',
            'summarization',
            'translation',
            'question-answering'
          ]}
          value={formValues.capabilities || []}
          onChange={(_, newValue) => handleChange('capabilities', newValue)}
          renderTags={(value, getTagProps) =>
            value.map((option, index) => (
              <Chip
                label={option}
                {...getTagProps({ index })}
                key={option}
              />
            ))
          }
          renderInput={(params) => (
            <TextField
              {...params}
              label="Capabilities"
              placeholder="Add capability"
              helperText="Agent capabilities"
            />
          )}
        />
      </Grid>
      
      {/* Parameters */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Temperature"
          name="temperature"
          type="number"
          inputProps={{ min: 0, max: 1, step: 0.1 }}
          value={formValues.parameters?.temperature || '0.7'}
          onChange={(e) => handleParameterChange('temperature', e.target.value)}
          helperText="Controls randomness (0-1)"
        />
      </Grid>
      
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Max Tokens"
          name="maxTokens"
          type="number"
          inputProps={{ min: 1, step: 1 }}
          value={formValues.parameters?.maxTokens || '1024'}
          onChange={(e) => handleParameterChange('maxTokens', e.target.value)}
          helperText="Maximum number of tokens to generate"
        />
      </Grid>
    </NodeFormBase>
  );
};

export default AgentNodeForm;
