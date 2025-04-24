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
import { Agent } from './AgentList';

// Define agent types
const AGENT_TYPES: FieldOption[] = [
  { value: 'ReactiveAgent', label: 'Reactive Agent' },
  { value: 'MemoryEnhancedAgent', label: 'Memory Enhanced Agent' },
  { value: 'UtilityBasedAgent', label: 'Utility Based Agent' },
  { value: 'RuleBasedAgent', label: 'Rule Based Agent' },
  // Learning agent types
  { value: 'MultiAgentRLAgent', label: 'Multi-Agent RL Agent' },
  { value: 'ModelBasedRLAgent', label: 'Model-Based RL Agent' }
];

// Define form values type
export interface AgentFormValues {
  name: string;
  description: string;
  agent_type: string;
  is_active: boolean;
  configuration: Record<string, any>;
}

// Define form errors type
interface FormErrors {
  name?: string;
  description?: string;
  agent_type?: string;
  configuration?: string;
}

interface AgentFormProps {
  initialValues?: Partial<AgentFormValues>;
  onSubmit: (values: AgentFormValues) => Promise<void>;
  isLoading?: boolean;
  error?: string | null;
  mode: 'create' | 'edit';
}

/**
 * AgentForm component for creating and editing agents.
 * Includes form validation and configuration based on agent type.
 */
const AgentForm: React.FC<AgentFormProps> = ({
  initialValues,
  onSubmit,
  isLoading = false,
  error = null,
  mode
}) => {
  const navigate = useNavigate();
  const [values, setValues] = useState<AgentFormValues>({
    name: '',
    description: '',
    agent_type: 'ReactiveAgent',
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

    // Reset configuration when agent type changes
    if (name === 'agent_type') {
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

    // Validate agent type
    if (!values.agent_type) {
      newErrors.agent_type = 'Agent type is required';
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
      navigate('/agents');
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  // Handle cancel
  const handleCancel = () => {
    navigate('/agents');
  };

  // Render configuration fields based on agent type
  const renderConfigurationFields = () => {
    switch (values.agent_type) {
      case 'ReactiveAgent':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="number"
                name="max_iterations"
                label="Max Iterations"
                value={values.configuration.max_iterations || 10}
                onChange={handleConfigChange}
                min={1}
                max={100}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="prompt_template"
                label="Prompt Template"
                value={values.configuration.prompt_template || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="Template for the agent's prompt. Use {input} as a placeholder for user input."
              />
            </Grid>
          </>
        );
        
      case 'MemoryEnhancedAgent':
        return (
          <>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="memory_size"
                label="Memory Size"
                value={values.configuration.memory_size || 5}
                onChange={handleConfigChange}
                min={1}
                max={50}
                helperText="Number of past interactions to remember"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="memory_type"
                label="Memory Type"
                value={values.configuration.memory_type || 'episodic'}
                onChange={handleConfigChange}
                options={[
                  { value: 'episodic', label: 'Episodic' },
                  { value: 'semantic', label: 'Semantic' },
                  { value: 'working', label: 'Working' }
                ]}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="prompt_template"
                label="Prompt Template"
                value={values.configuration.prompt_template || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="Template for the agent's prompt. Use {input} as a placeholder for user input and {memory} for past interactions."
              />
            </Grid>
          </>
        );
        
      case 'UtilityBasedAgent':
        return (
          <>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="utility_threshold"
                label="Utility Threshold"
                value={values.configuration.utility_threshold || 0.5}
                onChange={handleConfigChange}
                min={0}
                max={1}
                step={0.1}
                helperText="Minimum utility score for action selection (0-1)"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="max_iterations"
                label="Max Iterations"
                value={values.configuration.max_iterations || 10}
                onChange={handleConfigChange}
                min={1}
                max={100}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="utility_function"
                label="Utility Function"
                value={values.configuration.utility_function || ''}
                onChange={handleConfigChange}
                rows={5}
                helperText="JSON configuration for the utility function"
              />
            </Grid>
          </>
        );
        
      case 'RuleBasedAgent':
        return (
          <>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="rules"
                label="Rules"
                value={values.configuration.rules || ''}
                onChange={handleConfigChange}
                rows={8}
                helperText="JSON array of rules. Each rule should have a condition and an action."
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="rule_evaluation"
                label="Rule Evaluation"
                value={values.configuration.rule_evaluation || 'first_match'}
                onChange={handleConfigChange}
                options={[
                  { value: 'first_match', label: 'First Match' },
                  { value: 'all_matching', label: 'All Matching' },
                  { value: 'priority', label: 'Priority Based' }
                ]}
              />
            </Grid>
          </>
        );
        
      case 'MultiAgentRLAgent':
        return (
          <>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="num_agents"
                label="Number of Agents"
                value={values.configuration.num_agents || 2}
                onChange={handleConfigChange}
                min={1}
                max={10}
                helperText="Number of agents in the multi-agent system"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="select"
                name="cooperative"
                label="Cooperation Mode"
                value={values.configuration.cooperative === false ? false : true}
                onChange={handleConfigChange}
                options={[
                  { value: true, label: 'Cooperative' },
                  { value: false, label: 'Competitive' }
                ]}
                helperText="Whether agents cooperate or compete"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="learning_rate"
                label="Learning Rate"
                value={values.configuration.learning_rate || 0.001}
                onChange={handleConfigChange}
                min={0.0001}
                max={0.1}
                step={0.0001}
                helperText="Learning rate for neural network training"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="hidden_dim"
                label="Hidden Dimension"
                value={values.configuration.hidden_dim || 128}
                onChange={handleConfigChange}
                min={16}
                max={512}
                step={16}
                helperText="Size of hidden layers in neural networks"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="discount_factor"
                label="Discount Factor"
                value={values.configuration.discount_factor || 0.99}
                onChange={handleConfigChange}
                min={0.8}
                max={0.999}
                step={0.001}
                helperText="Discount factor for future rewards (0.8-0.999)"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="memory_size"
                label="Memory Size"
                value={values.configuration.memory_size || 10000}
                onChange={handleConfigChange}
                min={1000}
                max={100000}
                step={1000}
                helperText="Size of replay memory buffer"
              />
            </Grid>
          </>
        );
        
      case 'ModelBasedRLAgent':
        return (
          <>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="state_dim"
                label="State Dimension"
                value={values.configuration.state_dim || 4}
                onChange={handleConfigChange}
                min={1}
                max={100}
                helperText="Dimension of the state space"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="action_dim"
                label="Action Dimension"
                value={values.configuration.action_dim || 2}
                onChange={handleConfigChange}
                min={1}
                max={50}
                helperText="Dimension of the action space"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="planning_horizon"
                label="Planning Horizon"
                value={values.configuration.planning_horizon || 5}
                onChange={handleConfigChange}
                min={1}
                max={20}
                helperText="How many steps ahead to plan"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="num_simulations"
                label="Number of Simulations"
                value={values.configuration.num_simulations || 10}
                onChange={handleConfigChange}
                min={1}
                max={100}
                helperText="How many simulations to run for planning"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="learning_rate"
                label="Learning Rate"
                value={values.configuration.learning_rate || 0.001}
                onChange={handleConfigChange}
                min={0.0001}
                max={0.1}
                step={0.0001}
                helperText="Learning rate for neural network training"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormField
                type="number"
                name="discount_factor"
                label="Discount Factor"
                value={values.configuration.discount_factor || 0.99}
                onChange={handleConfigChange}
                min={0.8}
                max={0.999}
                step={0.001}
                helperText="Discount factor for future rewards (0.8-0.999)"
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
              label="Agent Name"
              value={values.name}
              onChange={handleChange}
              error={touched.name ? errors.name : undefined}
              required
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="agent_type"
              label="Agent Type"
              value={values.agent_type}
              onChange={handleChange}
              options={AGENT_TYPES}
              error={touched.agent_type ? errors.agent_type : undefined}
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
                {mode === 'create' ? 'Create Agent' : 'Update Agent'}
              </Button>
            </Box>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default AgentForm;
