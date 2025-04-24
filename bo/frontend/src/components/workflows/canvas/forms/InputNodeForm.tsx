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
  Chip,
  Box
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';

interface InputNodeFormProps extends Omit<NodeFormProps, 'children'> {}

/**
 * Form for editing Input nodes
 */
const InputNodeForm: React.FC<InputNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    key: string;
    dataType: string;
    required: boolean;
    defaultValue?: string;
    options?: string[];
    validation?: string;
  }>({
    key: '',
    dataType: 'string',
    required: true,
    defaultValue: '',
    options: []
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        key: node.config.key || '',
        dataType: node.config.dataType || 'string',
        required: node.config.required !== false,
        defaultValue: node.config.defaultValue || '',
        options: node.config.options || [],
        validation: node.config.validation || ''
      });
    } else {
      setFormValues({
        key: '',
        dataType: 'string',
        required: true,
        defaultValue: '',
        options: []
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

  // Handle options change
  const handleOptionsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const optionsText = event.target.value;
    const optionsArray = optionsText.split(',').map(option => option.trim()).filter(Boolean);
    handleChange('options', optionsArray);
  };

  // Handle save
  const handleSave = (updatedNode: NodeData) => {
    const nodeWithConfig: NodeData = {
      ...updatedNode,
      config: {
        ...updatedNode.config,
        key: formValues.key,
        dataType: formValues.dataType,
        required: formValues.required,
        defaultValue: formValues.defaultValue,
        options: formValues.options,
        validation: formValues.validation
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
      {/* Input Key */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Input Key"
          name="key"
          value={formValues.key || ''}
          onChange={(e) => handleChange('key', e.target.value)}
          helperText="Unique identifier for this input"
          required
        />
      </Grid>
      
      {/* Data Type */}
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth>
          <InputLabel id="dataType-select-label">Data Type</InputLabel>
          <Select
            labelId="dataType-select-label"
            id="dataType-select"
            value={formValues.dataType || 'string'}
            label="Data Type"
            onChange={(e) => handleChange('dataType', e.target.value)}
          >
            <MenuItem value="string">String</MenuItem>
            <MenuItem value="number">Number</MenuItem>
            <MenuItem value="boolean">Boolean</MenuItem>
            <MenuItem value="object">Object</MenuItem>
            <MenuItem value="array">Array</MenuItem>
            <MenuItem value="date">Date</MenuItem>
            <MenuItem value="file">File</MenuItem>
          </Select>
          <FormHelperText>Type of data expected</FormHelperText>
        </FormControl>
      </Grid>
      
      {/* Required */}
      <Grid item xs={12} sm={6}>
        <FormControlLabel
          control={
            <Switch
              checked={formValues.required}
              onChange={(e) => handleChange('required', e.target.checked)}
              name="required"
            />
          }
          label="Required"
        />
      </Grid>
      
      {/* Default Value */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Default Value"
          name="defaultValue"
          value={formValues.defaultValue || ''}
          onChange={(e) => handleChange('defaultValue', e.target.value)}
          helperText="Default value if none is provided"
        />
      </Grid>
      
      {/* Options (for select/dropdown inputs) */}
      {(formValues.dataType === 'string' || formValues.dataType === 'number') && (
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Options"
            name="options"
            value={(formValues.options || []).join(', ')}
            onChange={handleOptionsChange}
            helperText="Comma-separated list of allowed values (leave empty for any value)"
          />
          
          {formValues.options && formValues.options.length > 0 && (
            <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {formValues.options.map((option, index) => (
                <Chip 
                  key={index} 
                  label={option} 
                  size="small" 
                  onDelete={() => {
                    const newOptions = [...formValues.options || []];
                    newOptions.splice(index, 1);
                    handleChange('options', newOptions);
                  }}
                />
              ))}
            </Box>
          )}
        </Grid>
      )}
      
      {/* Validation */}
      <Grid item xs={12}>
        <TextField
          fullWidth
          label="Validation"
          name="validation"
          value={formValues.validation || ''}
          onChange={(e) => handleChange('validation', e.target.value)}
          helperText="Validation expression (e.g., for string: ^[A-Za-z]+$)"
        />
      </Grid>
    </NodeFormBase>
  );
};

export default InputNodeForm;
