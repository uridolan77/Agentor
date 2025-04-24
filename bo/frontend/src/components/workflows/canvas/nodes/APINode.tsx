import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Tooltip, Chip } from '@mui/material';
import HttpIcon from '@mui/icons-material/Http';
import { NODE_TYPE_CONFIG, NodeType } from '../types/nodeTypes';

/**
 * API Node component for the workflow canvas.
 * This node enables API operations in workflows.
 */
const APINode: React.FC<NodeProps> = ({ data, selected }) => {
  const { label, config } = data;
  const nodeConfig = NODE_TYPE_CONFIG[NodeType.API];

  // Get method color
  const getMethodColor = (method: string): string => {
    switch (method) {
      case 'GET':
        return '#4caf50'; // Green
      case 'POST':
        return '#2196f3'; // Blue
      case 'PUT':
        return '#ff9800'; // Orange
      case 'DELETE':
        return '#f44336'; // Red
      case 'PATCH':
        return '#9c27b0'; // Purple
      default:
        return '#757575'; // Grey
    }
  };

  const method = config.method || 'GET';
  const urlTemplate = config.url_template || '';
  const shortUrl = urlTemplate.length > 25 
    ? `...${urlTemplate.substring(urlTemplate.length - 25)}` 
    : urlTemplate;

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
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input-request"
        style={{ background: nodeConfig.color }}
      />

      {/* Node Content */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <HttpIcon sx={{ mr: 1, color: nodeConfig.color }} />
        <Typography variant="subtitle1" noWrap>
          {label}
        </Typography>
      </Box>

      <Tooltip title={`${method} ${urlTemplate}`}>
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={method} 
            size="small" 
            sx={{ 
              bgcolor: `${getMethodColor(method)}33`,
              color: getMethodColor(method),
              fontWeight: 'bold',
              fontSize: '0.7rem'
            }} 
          />
          <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.5 }}>
            URL: {shortUrl || '(not set)'}
          </Typography>
        </Box>
      </Tooltip>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output-response"
        style={{ background: nodeConfig.color }}
      />
    </Box>
  );
};

export default APINode;
