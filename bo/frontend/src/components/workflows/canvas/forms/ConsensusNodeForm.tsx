import React from 'react';
import { Grid, Typography } from '@mui/material';
import { FormField } from '../../../common';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData } from '../types/nodeTypes';

/**
 * Form component for configuring a Consensus node.
 * This node enables collective decision making among agents.
 */
const ConsensusNodeForm: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    consensus_method?: string;
    agreement_threshold?: number;
    max_rounds?: number;
    options_template?: string;
    use_weights?: boolean;
    store_voting_history?: boolean;
  }>({
    consensus_method: 'plurality',
    agreement_threshold: 0.75,
    max_rounds: 3,
    options_template: '',
    use_weights: false,
    store_voting_history: true
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        consensus_method: node.config.consensus_method || 'plurality',
        agreement_threshold: node.config.agreement_threshold || 0.75,
        max_rounds: node.config.max_rounds || 3,
        options_template: node.config.options_template || '',
        use_weights: node.config.use_weights || false,
        store_voting_history: node.config.store_voting_history !== false
      });
    } else {
      setFormValues({
        consensus_method: 'plurality',
        agreement_threshold: 0.75,
        max_rounds: 3,
        options_template: '',
        use_weights: false,
        store_voting_history: true
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
        consensus_method: formValues.consensus_method,
        agreement_threshold: formValues.agreement_threshold,
        max_rounds: formValues.max_rounds,
        options_template: formValues.options_template,
        use_weights: formValues.use_weights,
        store_voting_history: formValues.store_voting_history
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
        Consensus Configuration
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FormField
            type="select"
            name="consensus_method"
            label="Consensus Method"
            value={formValues.consensus_method || 'plurality'}
            onChange={handleChange}
            options={[
              { value: 'plurality', label: 'Plurality Voting' },
              { value: 'borda', label: 'Borda Count' },
              { value: 'runoff', label: 'Instant Runoff' },
              { value: 'delphi', label: 'Delphi Method' },
              { value: 'majority', label: 'Majority Rule' },
              { value: 'unanimity', label: 'Unanimity' }
            ]}
            helperText="Method used to reach consensus among agents"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="agreement_threshold"
            label="Agreement Threshold"
            value={formValues.agreement_threshold || 0.75}
            onChange={handleChange}
            min={0.5}
            max={1.0}
            step={0.05}
            helperText="Minimum percentage required for consensus (0.5-1.0)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="max_rounds"
            label="Maximum Rounds"
            value={formValues.max_rounds || 3}
            onChange={handleChange}
            min={1}
            max={10}
            helperText="Maximum number of voting rounds"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="textarea"
            name="options_template"
            label="Options Template"
            value={formValues.options_template || ''}
            onChange={handleChange}
            rows={4}
            helperText="Template for generating options (JSON format or leave empty for dynamic options)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="use_weights"
            label="Use Agent Weights"
            value={formValues.use_weights || false}
            onChange={handleChange}
            helperText="Weight agent votes based on confidence or authority"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="store_voting_history"
            label="Store Voting History"
            value={formValues.store_voting_history || true}
            onChange={handleChange}
            helperText="Keep record of all voting rounds"
          />
        </Grid>
      </Grid>
    </NodeFormBase>
  );
};

export default ConsensusNodeForm;
