import React, { memo, useState, useEffect } from 'react';
import { EdgeProps as ReactFlowEdgeProps, getBezierPath, EdgeLabelRenderer, useReactFlow } from 'reactflow';
import {
  Typography,
  Tooltip,
  IconButton,
  Paper,
  Box,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
  FormHelperText,
  Divider
} from '@mui/material';
import {
  Link as LinkIcon,
  LooksOne as LooksOneIcon,
  Filter1 as Filter1Icon,
  FilterNone as FilterNoneIcon,
  Edit as EditIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

interface RelationshipData {
  type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
  sourceColumn: string;
  targetColumn: string;
  updateEdge?: (id: string, data: any) => void;
}

interface ColumnOption {
  name: string;
  type: string;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
}

// Create a custom EdgeProps type that extends ReactFlow's EdgeProps
type EdgeProps = ReactFlowEdgeProps<RelationshipData>;

const RelationshipEdge: React.FC<EdgeProps> = ({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
  selected
}) => {
  const reactFlowInstance = useReactFlow();
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [sourceColumns, setSourceColumns] = useState<ColumnOption[]>([]);
  const [targetColumns, setTargetColumns] = useState<ColumnOption[]>([]);
  const [selectedSourceColumn, setSelectedSourceColumn] = useState(data?.sourceColumn || '');
  const [selectedTargetColumn, setSelectedTargetColumn] = useState(data?.targetColumn || '');
  const [selectedType, setSelectedType] = useState<'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many'>(data?.type || 'one-to-many');
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // Handle opening the relationship type menu
  const handleOpenMenu = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle closing the menu
  const handleCloseMenu = () => {
    setMenuAnchorEl(null);
  };

  // Load columns from source and target nodes
  useEffect(() => {
    if (reactFlowInstance) {
      const sourceNode = reactFlowInstance.getNode(source);
      const targetNode = reactFlowInstance.getNode(target);

      if (sourceNode?.data?.columns) {
        setSourceColumns(sourceNode.data.columns);

        // If we don't have a source column selected yet, try to find a primary key
        if (!data?.sourceColumn || data.sourceColumn === 'main') {
          const primaryKey = sourceNode.data.columns.find((col: ColumnOption) => col.isPrimaryKey);
          if (primaryKey && data?.updateEdge) {
            setSelectedSourceColumn(primaryKey.name);
            if (data.sourceColumn === 'main') { // Only auto-update if it's the default 'main'
              data.updateEdge(id, { sourceColumn: primaryKey.name });
            }
          }
        }
      }

      if (targetNode?.data?.columns) {
        setTargetColumns(targetNode.data.columns);

        // If we don't have a target column selected yet, try to find a primary key
        if (!data?.targetColumn || data.targetColumn === 'main') {
          const primaryKey = targetNode.data.columns.find((col: ColumnOption) => col.isPrimaryKey);
          if (primaryKey && data?.updateEdge) {
            setSelectedTargetColumn(primaryKey.name);
            if (data.targetColumn === 'main') { // Only auto-update if it's the default 'main'
              data.updateEdge(id, { targetColumn: primaryKey.name });
            }
          }
        }
      }
    }
  }, [reactFlowInstance, source, target, id, data]);

  // Handle changing the relationship type
  const handleChangeType = (type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many') => {
    if (data?.updateEdge) {
      data.updateEdge(id, { type });
    }
    handleCloseMenu();
  };

  // Handle opening the edit dialog
  const handleOpenEditDialog = () => {
    setSelectedSourceColumn(data?.sourceColumn || '');
    setSelectedTargetColumn(data?.targetColumn || '');
    setSelectedType(data?.type || 'one-to-many');
    setEditDialogOpen(true);
    handleCloseMenu();
  };

  // Handle closing the edit dialog
  const handleCloseEditDialog = () => {
    setEditDialogOpen(false);
  };

  // Handle saving the relationship changes
  const handleSaveRelationship = () => {
    if (data?.updateEdge) {
      data.updateEdge(id, {
        sourceColumn: selectedSourceColumn,
        targetColumn: selectedTargetColumn,
        type: selectedType
      });
    }
    handleCloseEditDialog();
  };

  // Handle source column change
  const handleSourceColumnChange = (event: SelectChangeEvent) => {
    setSelectedSourceColumn(event.target.value);
  };

  // Handle target column change
  const handleTargetColumnChange = (event: SelectChangeEvent) => {
    setSelectedTargetColumn(event.target.value);
  };

  // Handle relationship type change
  const handleTypeChange = (event: SelectChangeEvent) => {
    setSelectedType(event.target.value as 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many');
  };

  // Determine relationship icon based on type
  const getRelationshipIcon = () => {
    switch (data?.type) {
      case 'one-to-one':
        return <LooksOneIcon fontSize="small" />;
      case 'one-to-many':
        return <Filter1Icon fontSize="small" />;
      case 'many-to-one':
        return <Filter1Icon fontSize="small" sx={{ transform: 'rotate(180deg)' }} />;
      case 'many-to-many':
        return <FilterNoneIcon fontSize="small" />;
      default:
        return <LinkIcon fontSize="small" />;
    }
  };

  // Format relationship label text
  const getRelationshipLabel = () => {
    switch (data?.type) {
      case 'one-to-one':
        return '1:1';
      case 'one-to-many':
        return '1:N';
      case 'many-to-one':
        return 'N:1';
      case 'many-to-many':
        return 'N:M';
      default:
        return '';
    }
  };

  return (
    <>
      <path
        id={id}
        style={{
          ...style,
          strokeWidth: selected ? 3 : 2,
          stroke: selected ? '#1976d2' : '#666',
        }}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />

      <EdgeLabelRenderer>
        <Box
          sx={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 12,
            pointerEvents: 'all',
          }}
        >
          <Paper
            elevation={selected ? 3 : 1}
            sx={{
              padding: '4px 8px',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              backgroundColor: selected ? 'rgba(25, 118, 210, 0.1)' : 'rgba(255, 255, 255, 0.9)',
              border: selected ? '1px solid #1976d2' : '1px solid #ddd',
            }}
          >
            {getRelationshipIcon()}
            <Typography variant="caption" sx={{ mx: 0.5 }}>
              {getRelationshipLabel()}
            </Typography>
            <Tooltip title={data?.sourceColumn === 'main' && data?.targetColumn === 'main' ?
              `${source} → ${target}` :
              `${source}.${data?.sourceColumn} → ${target}.${data?.targetColumn}`
            }>
              <Typography variant="caption" sx={{ mx: 0.5, color: 'text.secondary' }}>
                {data?.sourceColumn === 'main' && data?.targetColumn === 'main' ?
                  'Relationship' :
                  `${data?.sourceColumn} → ${data?.targetColumn}`
                }
              </Typography>
            </Tooltip>

            <Tooltip title="Change relationship type">
              <IconButton size="small" onClick={handleOpenMenu}>
                <EditIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            <Menu
              anchorEl={menuAnchorEl}
              open={Boolean(menuAnchorEl)}
              onClose={handleCloseMenu}
            >
              <MenuItem onClick={() => handleChangeType('one-to-one')}>
                <ListItemIcon>
                  <LooksOneIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="One-to-One (1:1)" />
              </MenuItem>
              <MenuItem onClick={() => handleChangeType('one-to-many')}>
                <ListItemIcon>
                  <Filter1Icon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="One-to-Many (1:N)" />
              </MenuItem>
              <MenuItem onClick={() => handleChangeType('many-to-one')}>
                <ListItemIcon>
                  <Filter1Icon fontSize="small" sx={{ transform: 'rotate(180deg)' }} />
                </ListItemIcon>
                <ListItemText primary="Many-to-One (N:1)" />
              </MenuItem>
              <MenuItem onClick={() => handleChangeType('many-to-many')}>
                <ListItemIcon>
                  <FilterNoneIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Many-to-Many (N:M)" />
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleOpenEditDialog}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Edit Connection Fields" />
              </MenuItem>
            </Menu>

            {/* Edit Relationship Dialog */}
            <Dialog open={editDialogOpen} onClose={handleCloseEditDialog} maxWidth="sm" fullWidth>
              <DialogTitle>Edit Relationship</DialogTitle>
              <DialogContent>
                <Box sx={{ mt: 2 }}>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="relationship-type-label">Relationship Type</InputLabel>
                    <Select
                      labelId="relationship-type-label"
                      value={selectedType}
                      label="Relationship Type"
                      onChange={handleTypeChange}
                    >
                      <MenuItem value="one-to-one">One-to-One (1:1)</MenuItem>
                      <MenuItem value="one-to-many">One-to-Many (1:N)</MenuItem>
                      <MenuItem value="many-to-one">Many-to-One (N:1)</MenuItem>
                      <MenuItem value="many-to-many">Many-to-Many (N:M)</MenuItem>
                    </Select>
                    <FormHelperText>Select the type of relationship between these tables</FormHelperText>
                  </FormControl>

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="source-column-label">Source Column ({source})</InputLabel>
                    <Select
                      labelId="source-column-label"
                      value={selectedSourceColumn}
                      label={`Source Column (${source})`}
                      onChange={handleSourceColumnChange}
                    >
                      <MenuItem value="main">Any Column</MenuItem>
                      {sourceColumns.map((column) => (
                        <MenuItem
                          key={column.name}
                          value={column.name}
                          sx={{
                            fontWeight: column.isPrimaryKey ? 'bold' : 'normal',
                            color: column.isPrimaryKey ? 'primary.main' : 'inherit'
                          }}
                        >
                          {column.name} {column.isPrimaryKey && '(PK)'} {column.isForeignKey && '(FK)'}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>Select the column from the source table</FormHelperText>
                  </FormControl>

                  <FormControl fullWidth>
                    <InputLabel id="target-column-label">Target Column ({target})</InputLabel>
                    <Select
                      labelId="target-column-label"
                      value={selectedTargetColumn}
                      label={`Target Column (${target})`}
                      onChange={handleTargetColumnChange}
                    >
                      <MenuItem value="main">Any Column</MenuItem>
                      {targetColumns.map((column) => (
                        <MenuItem
                          key={column.name}
                          value={column.name}
                          sx={{
                            fontWeight: column.isPrimaryKey ? 'bold' : 'normal',
                            color: column.isPrimaryKey ? 'primary.main' : 'inherit'
                          }}
                        >
                          {column.name} {column.isPrimaryKey && '(PK)'} {column.isForeignKey && '(FK)'}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>Select the column from the target table</FormHelperText>
                  </FormControl>
                </Box>
              </DialogContent>
              <DialogActions>
                <Button onClick={handleCloseEditDialog}>Cancel</Button>
                <Button onClick={handleSaveRelationship} variant="contained" color="primary">
                  Save
                </Button>
              </DialogActions>
            </Dialog>
          </Paper>
        </Box>
      </EdgeLabelRenderer>
    </>
  );
};

export default memo(RelationshipEdge);
