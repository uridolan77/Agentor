import React, { memo, useState, useRef, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  Collapse,
  Tooltip,
  Menu,
  MenuItem,
  alpha,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  TextField,
  InputAdornment
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  ContentCopy as DuplicateIcon,
  Edit as EditIcon,
  Key as KeyIcon,
  Link as LinkIcon,
  Storage as StorageIcon,
  DriveFileRenameOutline as RenameIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import { ColumnSchema } from '../../../../types/reporting';

interface TableNodeData {
  label: string;
  columns: ColumnSchema[];
  onNodeDelete?: (id: string) => void;
  onNodeUpdate?: (id: string, data: any) => void;
}

const TableNode: React.FC<NodeProps<TableNodeData>> = ({
  id,
  data,
  selected,
  isConnectable
}) => {
  const [expanded, setExpanded] = useState(false);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [newTableName, setNewTableName] = useState(data.label);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const menuOpen = Boolean(menuAnchorEl);

  // Handle menu open
  const handleMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  // Handle delete
  const handleDelete = () => {
    handleMenuClose();
    if (data.onNodeDelete) {
      data.onNodeDelete(id);
    }
  };

  // Start editing the table name
  const startEditing = () => {
    setNewTableName(data.label);
    setIsEditing(true);
    handleMenuClose();
    // Focus the input after it's rendered
    setTimeout(() => {
      if (nameInputRef.current) {
        nameInputRef.current.focus();
        nameInputRef.current.select();
      }
    }, 50);
  };

  // Stop editing and save changes
  const stopEditing = (save: boolean = true) => {
    if (save && data.onNodeUpdate && newTableName.trim() && newTableName !== data.label) {
      data.onNodeUpdate(id, { label: newTableName.trim() });
    }
    setIsEditing(false);
  };

  // Handle key press in the input field
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      stopEditing(true);
    } else if (e.key === 'Escape') {
      stopEditing(false);
    }
  };

  // Handle click outside to save changes
  useEffect(() => {
    if (isEditing) {
      const handleClickOutside = (e: MouseEvent) => {
        if (nameInputRef.current && !nameInputRef.current.contains(e.target as Node)) {
          stopEditing(true);
        }
      };

      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [isEditing]);

  // Node style based on selection state
  const nodeStyle = {
    minWidth: 250,
    maxWidth: 350,
    border: selected ? '2px solid #1976d2' : '1px solid #e0e0e0',
    borderRadius: 1,
    bgcolor: 'background.paper',
    position: 'relative',
    overflow: 'hidden'
  };

  // Get primary key columns
  const primaryKeys = data.columns.filter(col => col.isPrimaryKey);

  // Get foreign key columns
  const foreignKeys = data.columns.filter(col => col.isForeignKey);

  return (
    <>
      {/* Main connection handles - Left side */}
      <Tooltip title="Drag from here to connect to another table" placement="left">
        <div style={{ position: 'absolute', left: -8, top: '50%', transform: 'translateY(-50%)', zIndex: 5 }}>
          <Handle
            type="source"
            position={Position.Left}
            id={`${id}__main_left`}
            isConnectable={isConnectable}
            style={{
              width: 16,
              height: 16,
              background: '#1976d2',
              top: 0,
              left: 0,
              transform: 'none',
              border: '3px solid white',
              zIndex: 10,
              opacity: 1,
              boxShadow: '0 0 5px rgba(0,0,0,0.5)'
            }}
          />
        </div>
      </Tooltip>

      {/* Main connection handles - Right side */}
      <Tooltip title="Connect another table to this one" placement="right">
        <div style={{ position: 'absolute', right: -8, top: '50%', transform: 'translateY(-50%)', zIndex: 5 }}>
          <Handle
            type="target"
            position={Position.Right}
            id={`${id}__main_right`}
            isConnectable={isConnectable}
            style={{
              width: 16,
              height: 16,
              background: '#9c27b0',
              top: 0,
              right: 0,
              transform: 'none',
              border: '3px solid white',
              zIndex: 10,
              opacity: 1,
              boxShadow: '0 0 5px rgba(0,0,0,0.5)'
            }}
          />
        </div>
      </Tooltip>

      {/* Selection indicator */}
      {selected && (
        <Box
          sx={{
            position: 'absolute',
            top: -15,
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: '#1976d2',
            color: 'white',
            borderRadius: '4px',
            padding: '2px 8px',
            fontSize: '12px',
            fontWeight: 'bold',
            boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
            zIndex: 10
          }}
        >
          Selected
        </Box>
      )}

      <Paper
        elevation={selected ? 3 : 1}
        sx={nodeStyle}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Table Header */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            bgcolor: '#1976d2',
            color: 'white',
            px: 1.5,
            py: 1
          }}
        >
          <StorageIcon fontSize="small" sx={{ mr: 1 }} />
          {isEditing ? (
            <TextField
              inputRef={nameInputRef}
              value={newTableName}
              onChange={(e) => setNewTableName(e.target.value)}
              onKeyDown={handleKeyPress}
              onBlur={() => stopEditing(true)}
              size="small"
              variant="standard"
              autoFocus
              sx={{
                flexGrow: 1,
                '& .MuiInputBase-root': {
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1rem'
                },
                '& .MuiInput-underline:before': {
                  borderBottomColor: 'rgba(255, 255, 255, 0.5)'
                },
                '& .MuiInput-underline:hover:before': {
                  borderBottomColor: 'white'
                }
              }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      size="small"
                      onClick={() => stopEditing(true)}
                      sx={{ color: 'white' }}
                    >
                      <CheckIcon fontSize="small" />
                    </IconButton>
                  </InputAdornment>
                )
              }}
            />
          ) : (
            <Typography
              variant="subtitle1"
              sx={{
                flexGrow: 1,
                fontWeight: 'bold',
                cursor: 'pointer',
                '&:hover': {
                  textDecoration: 'underline'
                }
              }}
              onClick={startEditing}
            >
              {data.label}
            </Typography>
          )}

          <IconButton
            size="small"
            onClick={() => setExpanded(!expanded)}
            sx={{ color: 'white', mr: 0.5 }}
          >
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>

          <IconButton
            size="small"
            onClick={handleMenuOpen}
            sx={{ color: 'white' }}
          >
            <MoreVertIcon />
          </IconButton>

          <Menu
            anchorEl={menuAnchorEl}
            open={menuOpen}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={startEditing} dense>
              <RenameIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
              Rename
            </MenuItem>
            <MenuItem onClick={handleDelete} dense>
              <DeleteIcon fontSize="small" sx={{ mr: 1, color: 'error.main' }} />
              Remove
            </MenuItem>
          </Menu>

          {/* We've replaced the dialog with inline editing */}
        </Box>

        {/* Table Summary (when collapsed) */}
        {!expanded && (
          <Box sx={{ p: 1.5 }}>
            <Typography variant="body2" color="text.secondary">
              {data.columns.length} columns
            </Typography>

            {primaryKeys.length > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                <KeyIcon fontSize="small" color="primary" sx={{ mr: 0.5 }} />
                <Typography variant="body2">
                  PK: {primaryKeys.map(pk => pk.name).join(', ')}
                </Typography>
              </Box>
            )}

            {foreignKeys.length > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                <LinkIcon fontSize="small" color="secondary" sx={{ mr: 0.5 }} />
                <Typography variant="body2">
                  FK: {foreignKeys.length} relations
                </Typography>
              </Box>
            )}
          </Box>
        )}

        {/* Table Columns (when expanded) */}
        <Collapse in={expanded}>
          <List dense disablePadding>
            <Divider />
            {data.columns.map((column, index) => (
              <React.Fragment key={`${id}-${column.name}`}>
                <ListItem
                  sx={{
                    px: 1.5,
                    py: 0.75,
                    bgcolor: column.isPrimaryKey ? alpha('#1976d2', 0.08) : 'inherit'
                  }}
                >
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {column.isPrimaryKey && (
                          <Tooltip title="Primary Key">
                            <KeyIcon fontSize="small" color="primary" sx={{ mr: 0.5 }} />
                          </Tooltip>
                        )}
                        {column.isForeignKey && (
                          <Tooltip title={`Foreign Key to ${column.references?.table}.${column.references?.column}`}>
                            <LinkIcon fontSize="small" color="secondary" sx={{ mr: 0.5 }} />
                          </Tooltip>
                        )}
                        <Typography variant="body2" sx={{ fontWeight: column.isPrimaryKey ? 'bold' : 'normal' }}>
                          {column.name}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Typography variant="caption" color="text.secondary">
                        {column.type}
                      </Typography>
                    }
                  />

                  {/* Handle for each column */}
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={`${id}__${column.name}`}
                    isConnectable={isConnectable}
                    style={{
                      width: 12,
                      height: 12,
                      background: column.isPrimaryKey ? '#1976d2' : '#666',
                      right: -6,
                      border: '2px solid white',
                      zIndex: 10,
                      opacity: 0.8,
                      boxShadow: '0 0 3px rgba(0,0,0,0.5)'
                    }}
                  />

                  <Handle
                    type="target"
                    position={Position.Left}
                    id={`${id}__${column.name}`}
                    isConnectable={isConnectable}
                    style={{
                      width: 12,
                      height: 12,
                      background: column.isForeignKey ? '#9c27b0' : '#666',
                      left: -6,
                      border: '2px solid white',
                      zIndex: 10,
                      opacity: 0.8,
                      boxShadow: '0 0 3px rgba(0,0,0,0.5)'
                    }}
                  />
                </ListItem>
                {index < data.columns.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Collapse>
      </Paper>
    </>
  );
};

export default memo(TableNode);
