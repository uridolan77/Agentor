import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Tooltip, Chip } from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import { NODE_TYPE_CONFIG, NodeType } from '../types/nodeTypes';

/**
 * Temporal Coordination Node component for the workflow canvas.
 * This node enables time-aware coordination among agents.
 */
const TemporalCoordinationNode: React.FC<NodeProps> = ({ data, selected }) => {
  const { label, config } = data;
  const nodeConfig = NODE_TYPE_CONFIG[NodeType.TEMPORAL_COORDINATION];

  // Get strategy display name
  const getStrategyDisplayName = (strategy: string): string => {
    switch (strategy) {
      case 'sequential':
        return 'Sequential Execution';
      case 'parallel':
        return 'Parallel Execution';
      case 'pipeline':
        return 'Pipeline Processing';
      case 'event_driven':
        return 'Event-Driven';
      case 'time_sliced':
        return 'Time-Sliced Execution';
      default:
        return 'Sequential Execution';
    }
  };

  // Get synchronization mode display name
  const getSyncModeDisplayName = (mode: string): string => {
    switch (mode) {
      case 'barrier':
        return 'Barrier Sync';
      case 'semaphore':
        return 'Semaphore';
      case 'lock':
        return 'Lock-based';
      case 'message_passing':
        return 'Message Passing';
      case 'token_ring':
        return 'Token Ring';
      default:
        return 'Barrier Sync';
    }
  };

  const strategy = config.coordination_strategy || 'sequential';
  const timeWindow = config.time_window || 60;
  const syncMode = config.synchronization_mode || 'barrier';

  return (
    <Box
      sx={{
        border: `2px solid ${nodeConfig.color}`,
        borderRadius: 2,
        p: 2,
        bgcolor: selected ? `${nodeConfig.color}22` : 'background.paper',
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Input Handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="input-agents"
        style={{ top: '30%', background: nodeConfig.color }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="input-task"
        style={{ top: '70%', background: nodeConfig.color }}
      />

      {/* Node Content */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <AccessTimeIcon sx={{ mr: 1, color: nodeConfig.color }} />
        <Typography variant="subtitle1" noWrap>
          {label}
        </Typography>
      </Box>

      <Tooltip title={`Strategy: ${getStrategyDisplayName(strategy)}, Time Window: ${timeWindow}s, Sync Mode: ${getSyncModeDisplayName(syncMode)}`}>
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={getStrategyDisplayName(strategy)} 
            size="small" 
            sx={{ 
              bgcolor: `${nodeConfig.color}33`,
              color: 'text.primary',
              fontSize: '0.7rem'
            }} 
          />
          <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.5 }}>
            Window: {timeWindow}s | Mode: {getSyncModeDisplayName(syncMode)}
          </Typography>
        </Box>
      </Tooltip>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output-result"
        style={{ background: nodeConfig.color }}
      />
    </Box>
  );
};

export default TemporalCoordinationNode;
