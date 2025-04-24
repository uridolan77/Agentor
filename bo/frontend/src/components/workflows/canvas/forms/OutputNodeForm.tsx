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
  Radio,
  RadioGroup,
  FormLabel
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';

interface OutputNodeFormProps extends Omit<NodeFormProps, 'children'> {}

/**
 * Form for editing Output nodes
 */
const OutputNodeForm: React.FC<OutputNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    key: string;
    format: 'json' | 'text' | 'binary' | 'custom';
    customFormat?: string;
    transform?: string;
    cache?: boolean;
    cacheTtl?: number;
  }>({
    key: '',
    format: 'json',
    cache: false,
    cacheTtl: 3600
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        key: node.config.key || '',
        format: node.config.format || 'json',
        customFormat: node.config.customFormat || '',
        transform: node.config.transform || '',
        cache: node.config.cache || false,
        cacheTtl: node.config.cacheTtl || 3600
      });
    } else {
      setFormValues({
        key: '',
        format: 'json',
        cache: false,
        cacheTtl: 3600
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
        key: formValues.key,
        format: formValues.format,
        customFormat: formValues.customFormat,
        transform: formValues.transform,
        cache: formValues.cache,
        cacheTtl: formValues.cacheTtl
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
      {/* Output Key */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Output Key"
          name="key"
          value={formValues.key || ''}
          onChange={(e) => handleChange('key', e.target.value)}
          helperText="Unique identifier for this output"
          required
        />
      </Grid>
      
      {/* Output Format */}
      <Grid item xs={12}>
        <FormControl component="fieldset">
          <FormLabel component="legend">Output Format</FormLabel>
          <RadioGroup
            row
            name="format"
            value={formValues.format}
            onChange={(e) => handleChange('format', e.target.value)}
          >
            <FormControlLabel 
              value="json" 
              control={<Radio />} 
              label="JSON" 
            />
            <FormControlLabel 
              value="text" 
              control={<Radio />} 
              label="Text" 
            />
            <FormControlLabel 
              value="binary" 
              control={<Radio />} 
              label="Binary" 
            />
            <FormControlLabel 
              value="custom" 
              control={<Radio />} 
              label="Custom" 
            />
          </RadioGroup>
          <FormHelperText>Format of the output data</FormHelperText>
        </FormControl>
      </Grid>
      
      {/* Custom Format */}
      {formValues.format === 'custom' && (
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Custom Format"
            name="customFormat"
            value={formValues.customFormat || ''}
            onChange={(e) => handleChange('customFormat', e.target.value)}
            helperText="MIME type or format specification"
          />
        </Grid>
      )}
      
      {/* Transform */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Transform"
          name="transform"
          value={formValues.transform || ''}
          onChange={(e) => handleChange('transform', e.target.value)}
          helperText="JavaScript transformation function (leave empty for no transformation)"
          multiline
          rows={4}
        />
      </Grid>
      
      {/* Caching */}
      <Grid item xs={12}>
        <FormControlLabel
          control={
            <Switch
              checked={formValues.cache || false}
              onChange={(e) => handleChange('cache', e.target.checked)}
              name="cache"
            />
          }
          label="Cache Output"
        />
      </Grid>
      
      {/* Cache TTL */}
      {formValues.cache && (
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Cache TTL (seconds)"
            name="cacheTtl"
            type="number"
            inputProps={{ min: 1, step: 1 }}
            value={formValues.cacheTtl || 3600}
            onChange={(e) => handleChange('cacheTtl', parseInt(e.target.value))}
            helperText="Time to live for cached output (in seconds)"
          />
        </Grid>
      )}
    </NodeFormBase>
  );
};

export default OutputNodeForm;
