import React, { memo, useState, CSSProperties } from 'react';
import { Handle, Position, NodeProps, NodeResizer, NodeResizeControl } from 'reactflow';
import { 
  Paper, 
  Box, 
  Typography, 
  IconButton, 
  Collapse, 
  Tooltip,
  Menu,
  MenuItem,
  alpha
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  ContentCopy as DuplicateIcon,
  Edit as EditIcon,
  PlayArrow as PlayArrowIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { NodeData, NodeType, NODE_TYPE_CONFIG } from '../types';

export interface BaseNodeProps extends NodeProps<NodeData> {
  selected: boolean;
  onNodeEdit?: (id: string) => void;
  onNodeDelete?: (id: string) => void;
  onNodeDuplicate?: (id: string) => void;
  onNodeRun?: (id: string) => void;
}

/**
 * Base node component for all node types
 */
const BaseNode = ({ 
  id, 
  data, 
  selected, 
  onNodeEdit,
  onNodeDelete,
  onNodeDuplicate,
  onNodeRun,
  isConnectable,
  xPos,
  yPos
}: BaseNodeProps) => {
  const [expanded, setExpanded] = useState(true);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const menuOpen = Boolean(menuAnchorEl);
  
  // Get node type configuration
  const nodeConfig = NODE_TYPE_CONFIG[data.type];
  
  // Handle menu open
  const handleMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
  };
  
  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };
  
  // Toggle expanded state
  const toggleExpanded = (event: React.MouseEvent) => {
    event.stopPropagation();
    setExpanded(!expanded);
  };
  
  // Handle edit
  const handleEdit = (event: React.MouseEvent) => {
    event.stopPropagation();
    handleMenuClose();
    
    // Use the onNodeEdit from data if available, otherwise use the prop
    if (data.onNodeEdit) {
      data.onNodeEdit(id);
    } else if (onNodeEdit) {
      onNodeEdit(id);
    }
  };
  
  // Handle delete
  const handleDelete = (event: React.MouseEvent) => {
    event.stopPropagation();
    handleMenuClose();
    
    // Use the onNodeDelete from data if available, otherwise use the prop
    if (data.onNodeDelete) {
      data.onNodeDelete(id);
    } else if (onNodeDelete) {
      onNodeDelete(id);
    }
  };
  
  // Handle duplicate
  const handleDuplicate = (event: React.MouseEvent) => {
    event.stopPropagation();
    handleMenuClose();
    if (onNodeDuplicate) onNodeDuplicate(id);
  };
  
  // Handle run
  const handleRun = (event: React.MouseEvent) => {
    event.stopPropagation();
    handleMenuClose();
    if (onNodeRun) onNodeRun(id);
  };
  
  // Node style
  const nodeStyle: CSSProperties = {
    borderColor: selected ? nodeConfig.color : 'transparent',
    borderWidth: selected ? 3 : 0,
    borderStyle: selected ? 'solid' : 'none',
    minWidth: 200,
    minHeight: 100,
    boxShadow: selected ? `0 0 12px 4px ${alpha(nodeConfig.color, 0.7)}` : 'none',
    position: 'relative',
    ...data.style
  };
  
  return (
    <>
      <NodeResizer 
        minWidth={150}
        minHeight={80}
        isVisible={selected}
        lineClassName="noderesize-line"
        handleClassName="noderesize-handle"
        onResize={(event, { width, height }) => {
          if (data.size) {
            data.size.width = width;
            data.size.height = height;
          }
        }}
      />
      
      <Paper
        elevation={selected ? 3 : 1}
        sx={nodeStyle}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Selection indicator */}
        {selected && (
          <>
            {/* Top selection badge */}
            <Box
              sx={{
                position: 'absolute',
                top: -15,
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: nodeConfig.color,
                color: 'white',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '12px',
                fontWeight: 'bold',
                boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                zIndex: 10,
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <CheckCircleIcon fontSize="small" />
              Selected
            </Box>
            
            {/* Corner indicator dot */}
            <Box
              sx={{
                position: 'absolute',
                top: -10,
                right: -10,
                width: 20,
                height: 20,
                borderRadius: '50%',
                backgroundColor: nodeConfig.color,
                border: '2px solid white',
                boxShadow: '0 0 5px rgba(0,0,0,0.3)',
                zIndex: 10
              }}
            />
          </>
        )}
        {/* Node Header */}
        <Box
          sx={{
            bgcolor: nodeConfig.color,
            color: 'white',
            p: 1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderTopLeftRadius: 'inherit',
            borderTopRightRadius: 'inherit'
          }}
        >
          <Typography variant="subtitle2" noWrap sx={{ maxWidth: 140 }}>
            {data.label}
          </Typography>
          <Box display="flex" alignItems="center">
            <IconButton
              size="small"
              onClick={toggleExpanded}
              sx={{ color: 'white', p: 0.5 }}
            >
              {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
            <IconButton
              size="small"
              onClick={handleMenuOpen}
              sx={{ color: 'white', p: 0.5, ml: 0.5 }}
            >
              <MoreVertIcon fontSize="small" />
            </IconButton>
          </Box>
        </Box>
        
        {/* Node Content */}
        <Collapse in={expanded}>
          <Box sx={{ p: 1, minHeight: 40 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              component="div"
              sx={{
                display: 'inline-block',
                bgcolor: alpha(nodeConfig.color, 0.1),
                color: nodeConfig.color,
                px: 0.5,
                py: 0.25,
                borderRadius: 0.5,
                mb: 0.5
              }}
            >
              {nodeConfig.label}
            </Typography>
            
            {data.description && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {data.description}
              </Typography>
            )}
            
            {/* Custom content for specific node types */}
            <Box sx={{ mt: 1, minHeight: 20 }} className="node-content">
              {data.renderContent ? data.renderContent() : null}
            </Box>
          </Box>
        </Collapse>
        
        {/* Node Menu */}
        <Menu
          anchorEl={menuAnchorEl}
          open={menuOpen}
          onClose={handleMenuClose}
          onClick={(e) => e.stopPropagation()}
        >
          {(onNodeEdit || data.onNodeEdit) && (
            <MenuItem onClick={handleEdit}>
              <EditIcon fontSize="small" sx={{ mr: 1 }} />
              Edit
            </MenuItem>
          )}
          {onNodeDuplicate && (
            <MenuItem onClick={handleDuplicate}>
              <DuplicateIcon fontSize="small" sx={{ mr: 1 }} />
              Duplicate
            </MenuItem>
          )}
          {onNodeRun && (
            <MenuItem onClick={handleRun}>
              <PlayArrowIcon fontSize="small" sx={{ mr: 1 }} />
              Run
            </MenuItem>
          )}
          {(onNodeDelete || data.onNodeDelete) && (
            <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
              <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
              Delete
            </MenuItem>
          )}
        </Menu>
        
        {/* Input Handles */}
        {data.ports?.inputs.map((port, index) => {
          const portPosition = {
            top: `${((index + 1) / (data.ports!.inputs.length + 1)) * 100}%`,
            left: -8
          };
          
          return (
            <Tooltip key={`input-${port.id}`} title={port.label || port.type} placement="left">
              <Handle
                type="target"
                position={Position.Left}
                id={port.id}
                isConnectable={isConnectable}
                style={{
                  ...portPosition,
                  width: 12,
                  height: 12,
                  background: nodeConfig.color,
                  border: '2px solid white'
                }}
                data-type={port.type}
              />
            </Tooltip>
          );
        })}
        
        {/* Output Handles */}
        {data.ports?.outputs.map((port, index) => {
          const portPosition = {
            top: `${((index + 1) / (data.ports!.outputs.length + 1)) * 100}%`,
            right: -8
          };
          
          return (
            <Tooltip key={`output-${port.id}`} title={port.label || port.type} placement="right">
              <Handle
                type="source"
                position={Position.Right}
                id={port.id}
                isConnectable={isConnectable}
                style={{
                  ...portPosition,
                  width: 12,
                  height: 12,
                  background: nodeConfig.color,
                  border: '2px solid white'
                }}
                data-type={port.type}
              />
            </Tooltip>
          );
        })}
      </Paper>
    </>
  );
};

export default memo(BaseNode);
