import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Tooltip,
  Chip,
  useTheme
} from '@mui/material';
import {
  MoreVert as MoreIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ContentCopy as DuplicateIcon,
  PlayArrow as RunIcon
} from '@mui/icons-material';

// Node types with their corresponding colors
export const NODE_TYPES = {
  AGENT: { label: 'Agent', color: '#4caf50' },
  TOOL: { label: 'Tool', color: '#2196f3' },
  CONDITION: { label: 'Condition', color: '#ff9800' },
  INPUT: { label: 'Input', color: '#9c27b0' },
  OUTPUT: { label: 'Output', color: '#f44336' }
};

export interface NodeData {
  id: string;
  type: keyof typeof NODE_TYPES;
  label: string;
  description?: string;
  config?: Record<string, any>;
  position: { x: number; y: number };
}

interface WorkflowNodeProps {
  node: NodeData;
  selected: boolean;
  onSelect: (id: string) => void;
  onMove: (id: string, position: { x: number; y: number }) => void;
  onEdit: (id: string) => void;
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
  onRun: (id: string) => void;
}

/**
 * WorkflowNode component represents a node in the workflow editor.
 * It can be dragged, selected, and includes a context menu for actions.
 */
const WorkflowNode: React.FC<WorkflowNodeProps> = ({
  node,
  selected,
  onSelect,
  onMove,
  onEdit,
  onDelete,
  onDuplicate,
  onRun
}) => {
  const theme = useTheme();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Get node type configuration
  const nodeType = NODE_TYPES[node.type];

  // Handle node click
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(node.id);
  };

  // Handle menu open
  const handleMenuOpen = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setMenuAnchor(e.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  // Handle edit
  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    handleMenuClose();
    onEdit(node.id);
  };

  // Handle delete
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    handleMenuClose();
    onDelete(node.id);
  };

  // Handle duplicate
  const handleDuplicate = (e: React.MouseEvent) => {
    e.stopPropagation();
    handleMenuClose();
    onDuplicate(node.id);
  };

  // Handle run
  const handleRun = (e: React.MouseEvent) => {
    e.stopPropagation();
    handleMenuClose();
    onRun(node.id);
  };

  // Handle mouse down for dragging
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left mouse button
    e.stopPropagation();
    
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    
    setIsDragging(true);
    onSelect(node.id);
    
    // Add global event listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  // Handle mouse move for dragging
  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    // Calculate new position
    const canvasElement = document.getElementById('workflow-canvas');
    if (!canvasElement) return;
    
    const canvasRect = canvasElement.getBoundingClientRect();
    const x = e.clientX - canvasRect.left - dragOffset.x;
    const y = e.clientY - canvasRect.top - dragOffset.y;
    
    // Update node position
    onMove(node.id, { x, y });
  };

  // Handle mouse up to end dragging
  const handleMouseUp = () => {
    setIsDragging(false);
    
    // Remove global event listeners
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  return (
    <Paper
      sx={{
        position: 'absolute',
        left: `${node.position.x}px`,
        top: `${node.position.y}px`,
        width: 200,
        borderRadius: 1,
        overflow: 'hidden',
        boxShadow: selected ? 4 : 1,
        border: selected ? `2px solid ${theme.palette.primary.main}` : 'none',
        cursor: isDragging ? 'grabbing' : 'grab',
        zIndex: selected ? 10 : 1,
        transition: isDragging ? 'none' : 'box-shadow 0.2s, border 0.2s'
      }}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
    >
      {/* Node Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          bgcolor: nodeType.color,
          color: '#fff',
          p: 1
        }}
      >
        <Typography variant="subtitle2" noWrap sx={{ maxWidth: 140 }}>
          {node.label}
        </Typography>
        <IconButton
          size="small"
          onClick={handleMenuOpen}
          sx={{ color: '#fff', p: 0.5 }}
        >
          <MoreIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Node Content */}
      <Box sx={{ p: 1, bgcolor: 'background.paper' }}>
        <Chip
          label={nodeType.label}
          size="small"
          sx={{
            bgcolor: `${nodeType.color}20`,
            color: nodeType.color,
            mb: 1
          }}
        />
        {node.description && (
          <Typography variant="body2" color="text.secondary" noWrap>
            {node.description}
          </Typography>
        )}
      </Box>

      {/* Node Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleEdit}>
          <EditIcon fontSize="small" sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={handleDuplicate}>
          <DuplicateIcon fontSize="small" sx={{ mr: 1 }} />
          Duplicate
        </MenuItem>
        <MenuItem onClick={handleRun}>
          <RunIcon fontSize="small" sx={{ mr: 1 }} />
          Run
        </MenuItem>
        <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Connection Points */}
      <Tooltip title="Input">
        <Box
          sx={{
            position: 'absolute',
            left: -6,
            top: '50%',
            width: 12,
            height: 12,
            borderRadius: '50%',
            bgcolor: 'background.paper',
            border: `2px solid ${nodeType.color}`,
            transform: 'translateY(-50%)',
            cursor: 'pointer',
            zIndex: 2
          }}
          onClick={(e) => e.stopPropagation()}
        />
      </Tooltip>
      <Tooltip title="Output">
        <Box
          sx={{
            position: 'absolute',
            right: -6,
            top: '50%',
            width: 12,
            height: 12,
            borderRadius: '50%',
            bgcolor: 'background.paper',
            border: `2px solid ${nodeType.color}`,
            transform: 'translateY(-50%)',
            cursor: 'pointer',
            zIndex: 2
          }}
          onClick={(e) => e.stopPropagation()}
        />
      </Tooltip>
    </Paper>
  );
};

export default WorkflowNode;
