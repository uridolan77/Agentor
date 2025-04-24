import React, { useState, useRef, useEffect } from 'react';
import {
  Paper,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  TextField,
  InputAdornment,
  IconButton,
  Collapse,
  Menu,
  MenuItem,
  Tooltip
} from '@mui/material';
import {
  Storage as StorageIcon,
  Search as SearchIcon,
  Clear as ClearIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Edit as EditIcon,
  DriveFileRenameOutline as RenameIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import { TableSchema } from '../../../../types/reporting';

interface NodePaletteProps {
  tables: TableSchema[];
  readOnly?: boolean;
  onRenameTable?: (oldName: string, newName: string) => void;
}

/**
 * NodePalette component
 * Displays a palette of available tables that can be dragged onto the canvas
 */
const NodePalette: React.FC<NodePaletteProps> = ({ tables, readOnly = false, onRenameTable }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expanded, setExpanded] = useState(true);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedTable, setSelectedTable] = useState<TableSchema | null>(null);
  const [editingTableId, setEditingTableId] = useState<string | null>(null);
  const [newTableName, setNewTableName] = useState('');
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Filter tables based on search query
  const filteredTables = tables.filter(table =>
    table.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Handle drag start for table
  const onDragStart = (event: React.DragEvent<HTMLElement>, table: TableSchema) => {
    if (readOnly) return;

    event.dataTransfer.setData('application/reactflow/table', JSON.stringify(table));
    event.dataTransfer.effectAllowed = 'move';
  };

  // Handle opening the context menu
  const handleContextMenu = (event: React.MouseEvent<HTMLElement>, table: TableSchema) => {
    if (readOnly) return;

    event.preventDefault();
    event.stopPropagation();
    setSelectedTable(table);
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle closing the context menu
  const handleCloseMenu = () => {
    setMenuAnchorEl(null);
  };

  // Start editing a table name
  const startEditing = (table: TableSchema) => {
    setSelectedTable(table);
    setNewTableName(table.name);
    setEditingTableId(table.name);
    handleCloseMenu();

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
    if (save && selectedTable && onRenameTable && newTableName.trim() && newTableName !== selectedTable.name) {
      onRenameTable(selectedTable.name, newTableName.trim());
    }
    setEditingTableId(null);
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
    if (editingTableId) {
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
  }, [editingTableId]);

  return (
    <Paper
      elevation={2}
      sx={{
        width: 280,
        maxHeight: 'calc(100vh - 200px)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 1,
        bgcolor: 'background.paper',
        opacity: readOnly ? 0.7 : 1,
        ml: 0 // Remove margin to reduce gap
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          bgcolor: 'primary.main',
          color: 'primary.contrastText',
          px: 2,
          py: 1
        }}
      >
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
          Database Tables
        </Typography>
        <IconButton
          size="small"
          onClick={() => setExpanded(!expanded)}
          sx={{ color: 'inherit' }}
        >
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={expanded} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Search */}
        <Box sx={{ p: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder="Search tables..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: searchQuery && (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setSearchQuery('')}
                    edge="end"
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        </Box>

        <Divider />

        {/* Table List */}
        <List
          dense
          disablePadding
          sx={{
            overflowY: 'auto',
            flexGrow: 1,
            maxHeight: 'calc(100vh - 250px)'
          }}
        >
          {filteredTables.length === 0 ? (
            <ListItem>
              <ListItemText
                primary="No tables found"
                secondary={searchQuery ? "Try a different search term" : "Connect to a database with tables"}
              />
            </ListItem>
          ) : (
            filteredTables.map(table => (
              <ListItem
                key={table.name}
                draggable={!readOnly}
                onDragStart={(event) => onDragStart(event, table)}
                onContextMenu={(event) => handleContextMenu(event, table)}
                sx={{
                  cursor: readOnly ? 'default' : 'grab',
                  '&:hover': {
                    bgcolor: readOnly ? 'inherit' : 'action.hover'
                  }
                }}
              >
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <StorageIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {editingTableId === table.name ? (
                        <TextField
                          inputRef={nameInputRef}
                          value={newTableName}
                          onChange={(e) => setNewTableName(e.target.value)}
                          onKeyDown={handleKeyPress}
                          onBlur={() => stopEditing(true)}
                          size="small"
                          variant="standard"
                          autoFocus
                          onClick={(e) => e.stopPropagation()}
                          sx={{
                            minWidth: 120,
                            '& .MuiInputBase-root': {
                              fontSize: '0.875rem',
                              fontWeight: 'medium'
                            }
                          }}
                          InputProps={{
                            endAdornment: (
                              <InputAdornment position="end">
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    stopEditing(true);
                                  }}
                                >
                                  <CheckIcon fontSize="small" />
                                </IconButton>
                              </InputAdornment>
                            )
                          }}
                        />
                      ) : (
                        <Typography
                          variant="body2"
                          fontWeight="medium"
                          sx={{
                            cursor: 'pointer',
                            '&:hover': {
                              textDecoration: 'underline'
                            }
                          }}
                          onClick={(e) => {
                            if (!readOnly) {
                              e.stopPropagation();
                              startEditing(table);
                            }
                          }}
                        >
                          {table.name}
                        </Typography>
                      )}
                      {!readOnly && (
                        <Tooltip title="Rename table">
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              startEditing(table);
                            }}
                            sx={{ ml: 0.5, p: 0.5 }}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      )}
                    </Box>
                  }
                  secondary={`${table.columns.length} columns`}
                  secondaryTypographyProps={{
                    variant: 'caption'
                  }}
                />
              </ListItem>
            ))
          )}
        </List>

        {/* Instructions */}
        {!readOnly && (
          <>
            <Divider />
            <Box sx={{ p: 1, bgcolor: 'grey.100' }}>
              <Typography variant="caption" color="text.secondary" paragraph sx={{ mb: 0.5 }}>
                Drag tables onto the canvas to create your data model
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Connect tables using the handles on the sides of each table
              </Typography>
            </Box>
          </>
        )}

        {/* Context Menu */}
        <Menu
          anchorEl={menuAnchorEl}
          open={Boolean(menuAnchorEl)}
          onClose={handleCloseMenu}
        >
          <MenuItem onClick={() => selectedTable && startEditing(selectedTable)} dense>
            <ListItemIcon>
              <RenameIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Rename Table" />
          </MenuItem>
        </Menu>

        {/* We've replaced the dialog with inline editing */}
      </Collapse>
    </Paper>
  );
};

export default NodePalette;
