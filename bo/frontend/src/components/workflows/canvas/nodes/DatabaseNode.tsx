import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Box, Typography, Tooltip, Chip } from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import { NODE_TYPE_CONFIG, NodeType } from '../types/nodeTypes';

/**
 * Database Node component for the workflow canvas.
 * This node enables database operations in workflows.
 */
const DatabaseNode: React.FC<NodeProps> = ({ data, selected }) => {
  const { label, config } = data;
  const nodeConfig = NODE_TYPE_CONFIG[NodeType.DATABASE];

  // Get database type display name
  const getDatabaseTypeDisplay = (type: string): string => {
    switch (type) {
      case 'mysql':
        return 'MySQL';
      case 'postgresql':
        return 'PostgreSQL';
      case 'sqlite':
        return 'SQLite';
      case 'mongodb':
        return 'MongoDB';
      case 'oracle':
        return 'Oracle';
      case 'sqlserver':
        return 'SQL Server';
      default:
        return 'Database';
    }
  };

  // Get query type display name
  const getQueryTypeDisplay = (type: string): string => {
    switch (type) {
      case 'select':
        return 'SELECT';
      case 'insert':
        return 'INSERT';
      case 'update':
        return 'UPDATE';
      case 'delete':
        return 'DELETE';
      case 'procedure':
        return 'PROCEDURE';
      case 'custom':
        return 'CUSTOM';
      default:
        return 'QUERY';
    }
  };

  const databaseType = config.database_type || 'mysql';
  const queryType = config.query_type || 'select';

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
        id="input-query"
        style={{ background: nodeConfig.color }}
      />

      {/* Node Content */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <StorageIcon sx={{ mr: 1, color: nodeConfig.color }} />
        <Typography variant="subtitle1" noWrap>
          {label}
        </Typography>
      </Box>

      <Tooltip title={`${getDatabaseTypeDisplay(databaseType)} - ${getQueryTypeDisplay(queryType)}`}>
        <Box sx={{ mt: 1 }}>
          <Chip 
            label={getDatabaseTypeDisplay(databaseType)} 
            size="small" 
            sx={{ 
              bgcolor: `${nodeConfig.color}33`,
              color: 'text.primary',
              fontSize: '0.7rem'
            }} 
          />
          <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary', mt: 0.5 }}>
            Operation: {getQueryTypeDisplay(queryType)}
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

export default DatabaseNode;
