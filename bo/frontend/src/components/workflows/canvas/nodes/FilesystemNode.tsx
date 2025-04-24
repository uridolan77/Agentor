import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Tooltip, Chip } from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import { NODE_TYPE_CONFIG, NodeType } from '../types/nodeTypes';

/**
 * Filesystem Node component for the workflow canvas.
 * This node enables filesystem operations in workflows.
 */
const FilesystemNode: React.FC<NodeProps> = ({ data, selected }) => {
  const { label, config } = data;
  const nodeConfig = NODE_TYPE_CONFIG[NodeType.FILESYSTEM];

  // Get operation type display name
  const getOperationTypeDisplay = (type: string): string => {
    switch (type) {
      case 'read':
        return 'Read File';
      case 'write':
        return 'Write File';
      case 'append':
        return 'Append to File';
      case 'delete':
        return 'Delete File';
      case 'list':
        return 'List Directory';
      case 'exists':
        return 'Check if Exists';
      case 'copy':
        return 'Copy File';
      case 'move':
        return 'Move File';
      default:
        return 'File Operation';
    }
  };

  const operationType = config.operation_type || 'read';
  const pathTemplate = config.path_template || '';
  const shortPath = pathTemplate.length > 20 
    ? `...${pathTemplate.substring(pathTemplate.length - 20)}` 
    : pathTemplate;

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
        id="input-path"
        style={{ top: '30%', background: nodeConfig.color }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="input-operation"
        style={{ top: '70%', background: nodeConfig.color }}
      />

      {/* Node Content */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <FolderIcon sx={{ mr: 1, color: nodeConfig.color }} />
        <Typography variant="subtitle1" noWrap>
          {label}
        </Typography>
      </Box>

      <Tooltip title={`${getOperationTypeDisplay(operationType)} - ${pathTemplate}`}>
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={getOperationTypeDisplay(operationType)} 
            size="small" 
            sx={{ 
              bgcolor: `${nodeConfig.color}33`,
              color: 'text.primary',
              fontSize: '0.7rem'
            }} 
          />
          <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.5 }}>
            Path: {shortPath || '(not set)'}
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

export default FilesystemNode;
