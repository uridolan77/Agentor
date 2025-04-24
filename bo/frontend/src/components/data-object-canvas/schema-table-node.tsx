import React, { memo, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  IconButton,
  Collapse,
  Box,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon,
  DragIndicator as DragIndicatorIcon,
  Key as KeyIcon,
  Link as LinkIcon
} from '@mui/icons-material';

// Interface for column structure
interface ColumnSchema {
  name: string;
  type: string;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
  references?: {
    table: string;
    column: string;
  };
}

interface SchemaTableNodeData {
  label: string;
  columns: ColumnSchema[];
}

const SchemaTableNode: React.FC<NodeProps<SchemaTableNodeData>> = ({ data, id }) => {
  const [expanded, setExpanded] = useState(true);
  
  return (
    <Paper
      sx={{
        padding: 0,
        borderRadius: 1,
        width: 280,
        overflow: 'hidden',
        border: (theme) => `1px solid ${theme.palette.divider}`,
        boxShadow: 2
      }}
    >
      {/* Table Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          px: 2,
          py: 1,
          cursor: 'grab',
        }}
      >
        <DragIndicatorIcon fontSize="small" sx={{ mr: 1 }} />
        <Typography variant="subtitle1" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
          {data.label}
        </Typography>
        <IconButton 
          size="small" 
          onClick={() => setExpanded(!expanded)}
          sx={{ color: 'inherit' }}
        >
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      
      {/* Table Columns */}
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <List dense disablePadding>
          {data.columns.map((column, index) => (
            <React.Fragment key={`${id}-${column.name}`}>
              <ListItem
                sx={{ 
                  px: 2, 
                  py: 0.75,
                  bgcolor: column.isPrimaryKey ? 'rgba(25, 118, 210, 0.08)' : 'inherit'
                }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {column.isPrimaryKey && (
                        <Tooltip title="Primary Key">
                          <KeyIcon color="primary" fontSize="small" sx={{ mr: 0.5 }} />
                        </Tooltip>
                      )}
                      {column.isForeignKey && (
                        <Tooltip title="Foreign Key">
                          <LinkIcon color="secondary" fontSize="small" sx={{ mr: 0.5 }} />
                        </Tooltip>
                      )}
                      <Typography variant="body2">{column.name}</Typography>
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="text.secondary">
                      {column.type}
                    </Typography>
                  }
                  primaryTypographyProps={{ 
                    variant: 'body2'
                  }}
                />
                
                {/* Connection Handle for the column */}
                <Handle
                  id={`${id}__${column.name}`}
                  type="source"
                  position={Position.Right}
                  style={{ 
                    width: 10, 
                    height: 10, 
                    background: column.isPrimaryKey ? '#1976d2' : '#666',
                    right: -5
                  }}
                  isConnectable={true}
                />
                <Handle
                  id={`${id}__${column.name}`}
                  type="target"
                  position={Position.Left}
                  style={{ 
                    width: 10, 
                    height: 10, 
                    background: column.isForeignKey ? '#9c27b0' : '#666',
                    left: -5
                  }}
                  isConnectable={true}
                />
              </ListItem>
              {index < data.columns.length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))}
        </List>
      </Collapse>
      
      {/* Status footer - show column count when collapsed */}
      {!expanded && (
        <Box sx={{ p: 1, bgcolor: 'grey.100' }}>
          <Typography variant="caption" color="text.secondary">
            {data.columns.length} columns
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default memo(SchemaTableNode);