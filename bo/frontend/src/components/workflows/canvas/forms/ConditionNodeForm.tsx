import React from 'react';
import {
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormLabel
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';

interface ConditionNodeFormProps extends Omit<NodeFormProps, 'children'> {}

/**
 * Form for editing Condition nodes
 */
const ConditionNodeForm: React.FC<ConditionNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    conditionType: 'expression' | 'script' | 'comparison';
    expression?: string;
    script?: string;
    leftOperand?: string;
    rightOperand?: string;
    operator?: string;
    defaultPath?: 'true' | 'false';
  }>({
    conditionType: 'expression',
    expression: '',
    defaultPath: 'false'
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        conditionType: node.config.conditionType || 'expression',
        expression: node.config.expression || '',
        script: node.config.script || '',
        leftOperand: node.config.leftOperand || '',
        rightOperand: node.config.rightOperand || '',
        operator: node.config.operator || '==',
        defaultPath: node.config.defaultPath || 'false'
      });
    } else {
      setFormValues({
        conditionType: 'expression',
        expression: '',
        defaultPath: 'false'
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
        conditionType: formValues.conditionType,
        expression: formValues.expression,
        script: formValues.script,
        leftOperand: formValues.leftOperand,
        rightOperand: formValues.rightOperand,
        operator: formValues.operator,
        defaultPath: formValues.defaultPath
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
      {/* Condition Type */}
      <Grid item xs={12}>
        <FormControl component="fieldset">
          <FormLabel component="legend">Condition Type</FormLabel>
          <RadioGroup
            row
            name="conditionType"
            value={formValues.conditionType}
            onChange={(e) => handleChange('conditionType', e.target.value)}
          >
            <FormControlLabel 
              value="expression" 
              control={<Radio />} 
              label="Expression" 
            />
            <FormControlLabel 
              value="comparison" 
              control={<Radio />} 
              label="Comparison" 
            />
            <FormControlLabel 
              value="script" 
              control={<Radio />} 
              label="Script" 
            />
          </RadioGroup>
          <FormHelperText>Select how to define the condition</FormHelperText>
        </FormControl>
      </Grid>
      
      {/* Expression Condition */}
      {formValues.conditionType === 'expression' && (
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Expression"
            name="expression"
            value={formValues.expression || ''}
            onChange={(e) => handleChange('expression', e.target.value)}
            helperText="JavaScript expression that evaluates to true or false (e.g., data.value > 10)"
            multiline
            rows={3}
          />
        </Grid>
      )}
      
      {/* Comparison Condition */}
      {formValues.conditionType === 'comparison' && (
        <>
          <Grid item xs={12} sm={5}>
            <TextField
              fullWidth
              label="Left Operand"
              name="leftOperand"
              value={formValues.leftOperand || ''}
              onChange={(e) => handleChange('leftOperand', e.target.value)}
              helperText="Left side of comparison (e.g., data.value)"
            />
          </Grid>
          
          <Grid item xs={12} sm={2}>
            <FormControl fullWidth>
              <InputLabel id="operator-select-label">Operator</InputLabel>
              <Select
                labelId="operator-select-label"
                id="operator-select"
                value={formValues.operator || '=='}
                label="Operator"
                onChange={(e) => handleChange('operator', e.target.value)}
              >
                <MenuItem value="==">==</MenuItem>
                <MenuItem value="===">===</MenuItem>
                <MenuItem value="!=">!=</MenuItem>
                <MenuItem value="!==">!==</MenuItem>
                <MenuItem value=">">{'>'}</MenuItem>
                <MenuItem value=">=">{'≥'}</MenuItem>
                <MenuItem value="<">{'<'}</MenuItem>
                <MenuItem value="<=">{'≤'}</MenuItem>
                <MenuItem value="includes">includes</MenuItem>
                <MenuItem value="startsWith">starts with</MenuItem>
                <MenuItem value="endsWith">ends with</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={5}>
            <TextField
              fullWidth
              label="Right Operand"
              name="rightOperand"
              value={formValues.rightOperand || ''}
              onChange={(e) => handleChange('rightOperand', e.target.value)}
              helperText="Right side of comparison (e.g., 10)"
            />
          </Grid>
        </>
      )}
      
      {/* Script Condition */}
      {formValues.conditionType === 'script' && (
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Script"
            name="script"
            value={formValues.script || ''}
            onChange={(e) => handleChange('script', e.target.value)}
            helperText="JavaScript function that returns true or false"
            multiline
            rows={6}
          />
        </Grid>
      )}
      
      {/* Default Path */}
      <Grid item xs={12}>
        <FormControl component="fieldset">
          <FormLabel component="legend">Default Path</FormLabel>
          <RadioGroup
            row
            name="defaultPath"
            value={formValues.defaultPath}
            onChange={(e) => handleChange('defaultPath', e.target.value)}
          >
            <FormControlLabel 
              value="true" 
              control={<Radio />} 
              label="True" 
            />
            <FormControlLabel 
              value="false" 
              control={<Radio />} 
              label="False" 
            />
          </RadioGroup>
          <FormHelperText>Path to take if condition evaluation fails</FormHelperText>
        </FormControl>
      </Grid>
    </NodeFormBase>
  );
};

export default ConditionNodeForm;
