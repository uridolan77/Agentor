import React from 'react';
import { Grid, Typography } from '@mui/material';
import { FormField } from '../../../common';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData } from '../types/nodeTypes';

/**
 * Form component for configuring a Temporal Coordination node.
 * This node enables time-aware coordination among agents.
 */
const TemporalCoordinationNodeForm: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading
}) => {
  const [formValues, setFormValues] = React.useState<{
    coordination_strategy?: string;
    time_window?: number;
    max_wait_time?: number;
    synchronization_mode?: string;
    priority_based?: boolean;
    store_execution_history?: boolean;
  }>({
    coordination_strategy: 'sequential',
    time_window: 60,
    max_wait_time: 300,
    synchronization_mode: 'barrier',
    priority_based: false,
    store_execution_history: true
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        coordination_strategy: node.config.coordination_strategy || 'sequential',
        time_window: node.config.time_window || 60,
        max_wait_time: node.config.max_wait_time || 300,
        synchronization_mode: node.config.synchronization_mode || 'barrier',
        priority_based: node.config.priority_based || false,
        store_execution_history: node.config.store_execution_history !== false
      });
    } else {
      setFormValues({
        coordination_strategy: 'sequential',
        time_window: 60,
        max_wait_time: 300,
        synchronization_mode: 'barrier',
        priority_based: false,
        store_execution_history: true
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
        coordination_strategy: formValues.coordination_strategy,
        time_window: formValues.time_window,
        max_wait_time: formValues.max_wait_time,
        synchronization_mode: formValues.synchronization_mode,
        priority_based: formValues.priority_based,
        store_execution_history: formValues.store_execution_history
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
        Temporal Coordination Configuration
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12}>
          <FormField
            type="select"
            name="coordination_strategy"
            label="Coordination Strategy"
            value={formValues.coordination_strategy || 'sequential'}
            onChange={handleChange}
            options={[
              { value: 'sequential', label: 'Sequential Execution' },
              { value: 'parallel', label: 'Parallel Execution' },
              { value: 'pipeline', label: 'Pipeline Processing' },
              { value: 'event_driven', label: 'Event-Driven' },
              { value: 'time_sliced', label: 'Time-Sliced Execution' }
            ]}
            helperText="Strategy for coordinating agent execution over time"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="time_window"
            label="Time Window (seconds)"
            value={formValues.time_window || 60}
            onChange={handleChange}
            min={1}
            max={3600}
            helperText="Time window for coordination (1-3600 seconds)"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="number"
            name="max_wait_time"
            label="Maximum Wait Time (seconds)"
            value={formValues.max_wait_time || 300}
            onChange={handleChange}
            min={1}
            max={7200}
            helperText="Maximum time to wait for agent responses"
          />
        </Grid>

        <Grid item xs={12}>
          <FormField
            type="select"
            name="synchronization_mode"
            label="Synchronization Mode"
            value={formValues.synchronization_mode || 'barrier'}
            onChange={handleChange}
            options={[
              { value: 'barrier', label: 'Barrier Synchronization' },
              { value: 'semaphore', label: 'Semaphore' },
              { value: 'lock', label: 'Lock-based' },
              { value: 'message_passing', label: 'Message Passing' },
              { value: 'token_ring', label: 'Token Ring' }
            ]}
            helperText="Method used to synchronize agent execution"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="priority_based"
            label="Priority-Based Execution"
            value={formValues.priority_based || false}
            onChange={handleChange}
            helperText="Execute agents based on priority levels"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormField
            type="switch"
            name="store_execution_history"
            label="Store Execution History"
            value={formValues.store_execution_history || true}
            onChange={handleChange}
            helperText="Keep record of execution timeline"
          />
        </Grid>
      </Grid>
    </NodeFormBase>
  );
};

export default TemporalCoordinationNodeForm;
