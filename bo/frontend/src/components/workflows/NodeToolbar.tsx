import React from 'react';
import { Box, Paper, IconButton, Tooltip, useTheme } from '@mui/material';
import {
  SmartToy as AgentIcon,
  Build as ToolIcon,
  CallSplit as ConditionIcon,
  Input as InputIcon,
  Output as OutputIcon
} from '@mui/icons-material';
import { NODE_TYPES } from './WorkflowNode';

interface NodeToolbarProps {
  position: { x: number; y: number };
  onAddNode: (type: keyof typeof NODE_TYPES) => void;
}

/**
 * NodeToolbar component provides a toolbar for adding new nodes to the workflow.
 * It appears when a node is selected and allows adding connected nodes.
 */
const NodeToolbar: React.FC<NodeToolbarProps> = ({ position, onAddNode }) => {
  const theme = useTheme();

  return (
    <Paper
      sx={{
        position: 'absolute',
        left: `${position.x}px`,
        top: `${position.y}px`,
        display: 'flex',
        padding: 0.5,
        borderRadius: 2,
        boxShadow: 2,
        zIndex: 20,
        transform: 'translateX(-50%)'
      }}
    >
      <Tooltip title="Add Agent Node">
        <IconButton
          size="small"
          onClick={() => onAddNode('AGENT')}
          sx={{
            color: NODE_TYPES.AGENT.color,
            '&:hover': { bgcolor: `${NODE_TYPES.AGENT.color}10` }
          }}
        >
          <AgentIcon />
        </IconButton>
      </Tooltip>
      
      <Tooltip title="Add Tool Node">
        <IconButton
          size="small"
          onClick={() => onAddNode('TOOL')}
          sx={{
            color: NODE_TYPES.TOOL.color,
            '&:hover': { bgcolor: `${NODE_TYPES.TOOL.color}10` }
          }}
        >
          <ToolIcon />
        </IconButton>
      </Tooltip>
      
      <Tooltip title="Add Condition Node">
        <IconButton
          size="small"
          onClick={() => onAddNode('CONDITION')}
          sx={{
            color: NODE_TYPES.CONDITION.color,
            '&:hover': { bgcolor: `${NODE_TYPES.CONDITION.color}10` }
          }}
        >
          <ConditionIcon />
        </IconButton>
      </Tooltip>
      
      <Tooltip title="Add Input Node">
        <IconButton
          size="small"
          onClick={() => onAddNode('INPUT')}
          sx={{
            color: NODE_TYPES.INPUT.color,
            '&:hover': { bgcolor: `${NODE_TYPES.INPUT.color}10` }
          }}
        >
          <InputIcon />
        </IconButton>
      </Tooltip>
      
      <Tooltip title="Add Output Node">
        <IconButton
          size="small"
          onClick={() => onAddNode('OUTPUT')}
          sx={{
            color: NODE_TYPES.OUTPUT.color,
            '&:hover': { bgcolor: `${NODE_TYPES.OUTPUT.color}10` }
          }}
        >
          <OutputIcon />
        </IconButton>
      </Tooltip>
    </Paper>
  );
};

export default NodeToolbar;
