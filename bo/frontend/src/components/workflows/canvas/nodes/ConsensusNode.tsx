import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Tooltip, Chip } from '@mui/material';
import HowToVoteIcon from '@mui/icons-material/HowToVote';
import { BaseNodeProps } from './BaseNode';
import { NODE_TYPE_CONFIG, NodeType, NodeData } from '../types/nodeTypes';

/**
 * Consensus Node component for the workflow canvas.
 * This node enables collective decision making among agents.
 */
const ConsensusNode: React.FC<NodeProps> = ({ data, selected }) => {
  const { label, config } = data;
  const nodeConfig = NODE_TYPE_CONFIG[NodeType.CONSENSUS];

  // Get consensus method display name
  const getMethodDisplayName = (method: string): string => {
    switch (method) {
      case 'plurality':
        return 'Plurality Voting';
      case 'borda':
        return 'Borda Count';
      case 'runoff':
        return 'Instant Runoff';
      case 'delphi':
        return 'Delphi Method';
      case 'majority':
        return 'Majority Rule';
      case 'unanimity':
        return 'Unanimity';
      default:
        return 'Plurality Voting';
    }
  };

  const consensusMethod = config.consensus_method || 'plurality';
  const threshold = config.agreement_threshold || 0.75;
  const maxRounds = config.max_rounds || 3;

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
        id="input-options"
        style={{ top: '70%', background: nodeConfig.color }}
      />

      {/* Node Content */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <HowToVoteIcon sx={{ mr: 1, color: nodeConfig.color }} />
        <Typography variant="subtitle1" noWrap>
          {label}
        </Typography>
      </Box>

      <Tooltip title={`Method: ${getMethodDisplayName(consensusMethod)}, Threshold: ${threshold * 100}%, Max Rounds: ${maxRounds}`}>
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={getMethodDisplayName(consensusMethod)} 
            size="small" 
            sx={{ 
              bgcolor: `${nodeConfig.color}33`,
              color: 'text.primary',
              fontSize: '0.7rem'
            }} 
          />
          <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.5 }}>
            Threshold: {threshold * 100}% | Rounds: {maxRounds}
          </Typography>
        </Box>
      </Tooltip>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output-decision"
        style={{ background: nodeConfig.color }}
      />
    </Box>
  );
};

export default ConsensusNode;
