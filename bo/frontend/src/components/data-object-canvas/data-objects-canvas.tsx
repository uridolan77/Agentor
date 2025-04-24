// filepath: d:\code\agentor\bo\frontend\src\components\data-object-canvas\data-objects-canvas.tsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  NodeTypes,
  EdgeTypes,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';

import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  ListItemIcon,
  Divider,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Chip,
  Alert,
  Drawer,
  AppBar,
  Toolbar,
  IconButton,
  CircularProgress,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardHeader,
  Collapse,
  Grid,
  Switch,
  FormControlLabel,
  Tooltip,
  Snackbar
} from '@mui/material';

import {
  TableView as TableViewIcon,
  Link as LinkIcon,
  Save as SaveIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  FilterList as FilterListIcon,
  ArrowBack as ArrowBackIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Search as SearchIcon,
  AutoGraph as AutoGraphIcon
} from '@mui/icons-material';

import useDataSourceStore from '../../store/dataSourceStore';
import SchemaTableNode from './schema-table-node';
import RelationshipEdge from './relationship-edge';
import { DataSource } from '../../types/reporting';
import * as dataObjectsApi from './data-objects-api';

// Define node types for the ReactFlow canvas
const nodeTypes: NodeTypes = {
  schemaTable: SchemaTableNode
};

// Define edge types for the ReactFlow canvas
const edgeTypes: EdgeTypes = {
  relationship: RelationshipEdge
};

// Interface for table structure
interface TableSchema {
  name: string;
  columns: ColumnSchema[];
}

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

// Interface for the relationships
interface Relationship {
  id: string;
  sourceTable: string;
  sourceColumn: string;
  targetTable: string;
  targetColumn: string;
  type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
}

// Interface for saved data model
interface DataModel {
  id: string;
  name: string;
  dataSourceId: string;
  tables: string[];
  relationships: Relationship[];
  layout?: string; // JSON string containing layout information
  createdAt: string;
  updatedAt: string;
}

const DataObjectsCanvas: React.FC = () => {
  // Reference to the data source store
  const {
    dataSources,
    selectedDataSource,
    selectDataSource,
    fetchDataSources,
    loading: dataSourceLoading
  } = useDataSourceStore();

  // State for the ReactFlow nodes and edges
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // State for database tables and selected tables
  const [tables, setTables] = useState<TableSchema[]>([]);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [expandedTable, setExpandedTable] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // State for the data model
  const [dataModel, setDataModel] = useState<DataModel | null>(null);
  const [dataModelName, setDataModelName] = useState('New Data Model');
  const [savedModels, setSavedModels] = useState<DataModel[]>([]);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [showTableDrawer, setShowTableDrawer] = useState(true);
  const [drawerWidth, setDrawerWidth] = useState(320);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Dialogs state
  const [openRelationshipDialog, setOpenRelationshipDialog] = useState(false);
  const [openSaveModelDialog, setOpenSaveModelDialog] = useState(false);
  const [openLoadModelDialog, setOpenLoadModelDialog] = useState(false);

  // Relationship dialog state
  const [relationshipForm, setRelationshipForm] = useState({
    sourceTable: '',
    sourceColumn: '',
    targetTable: '',
    targetColumn: '',
    type: 'one-to-many' as 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many'
  });

  // Reference to the ReactFlow instance
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Fetch database tables when data source changes
  useEffect(() => {
    if (selectedDataSource) {
      fetchTableSchemas(selectedDataSource.id);
    } else if (dataSources.length > 0) {
      selectDataSource(dataSources[0].id);
    } else {
      fetchDataSources();
    }
  }, [selectedDataSource, dataSources, selectDataSource, fetchDataSources]);

  // Fetch tables from the selected data source
  const fetchTableSchemas = async (dataSourceId: string) => {
    try {
      setLoading(true);
      setError(null);

      // Call the API to get table schemas
      const response = await dataObjectsApi.getDatabaseSchema(dataSourceId);

      if (Array.isArray(response) && response.length > 0) {
        setTables(response);
        setNotification({
          open: true,
          message: `Successfully loaded ${response.length} tables from the data source`,
          severity: 'success'
        });
      } else if (Array.isArray(response) && response.length === 0) {
        setTables([]);
        setNotification({
          open: true,
          message: 'No tables found in the selected data source',
          severity: 'info'
        });
      } else {
        throw new Error('Invalid response format from server');
      }

      setLoading(false);
    } catch (err) {
      console.error('Error fetching table schemas:', err);
      setError(`Failed to fetch database schema: ${err instanceof Error ? err.message : 'Unknown error'}. Please check your connection and try again.`);
      setNotification({
        open: true,
        message: `Error loading tables: ${err instanceof Error ? err.message : 'Unknown error'}`,
        severity: 'error'
      });
      setLoading(false);
      setTables([]);
    }
  };

  // Fetch saved data models
  const fetchSavedModels = async () => {
    try {
      if (!selectedDataSource) return;

      setLoading(true);
      const models = await dataObjectsApi.getDataModels(selectedDataSource.id);
      setSavedModels(models);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching data models:', err);
      setError('Failed to fetch saved data models.');
      setLoading(false);
    }
  };

  // Handle edge creation (table relationships)
  const onConnect = useCallback((connection: Connection) => {
    // Extract source and target table/column from connection
    const sourceId = connection.source?.split('__')[0];
    const targetId = connection.target?.split('__')[0];
    const sourceColumn = connection.source?.split('__')[1];
    const targetColumn = connection.target?.split('__')[1];

    if (sourceId && targetId && sourceColumn && targetColumn) {
      setRelationshipForm({
        sourceTable: sourceId,
        sourceColumn: sourceColumn,
        targetTable: targetId,
        targetColumn: targetColumn,
        type: 'one-to-many'
      });

      setOpenRelationshipDialog(true);
    }
  }, []);

  // Create a relationship between tables
  const createRelationship = () => {
    const { sourceTable, sourceColumn, targetTable, targetColumn, type } = relationshipForm;

    // Create a unique ID for the edge
    const edgeId = `${sourceTable}__${sourceColumn}-${targetTable}__${targetColumn}`;

    // Add the edge to the graph
    const newEdge: Edge = {
      id: edgeId,
      source: sourceTable,
      target: targetTable,
      sourceHandle: `${sourceTable}__${sourceColumn}`,
      targetHandle: `${targetTable}__${targetColumn}`,
      type: 'relationship',
      animated: true,
      data: {
        type: type,
        sourceColumn,
        targetColumn
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 20,
        height: 20
      }
    };

    setEdges((prevEdges) => [...prevEdges, newEdge]);
    setOpenRelationshipDialog(false);
  };

  // Add a table to the canvas
  const addTableToCanvas = (table: TableSchema) => {
    // Check if table is already added
    if (selectedTables.includes(table.name)) {
      return;
    }

    // Calculate position based on the number of nodes
    const position = {
      x: 250 + (nodes.length % 3) * 300,
      y: 100 + Math.floor(nodes.length / 3) * 400
    };

    // Create node for the table
    const newNode: Node = {
      id: table.name,
      type: 'schemaTable',
      position,
      data: {
        label: table.name,
        columns: table.columns
      }
    };

    // Update states
    setNodes((prevNodes) => [...prevNodes, newNode]);
    setSelectedTables((prevTables) => [...prevTables, table.name]);
  };

  // Remove a table from the canvas
  const removeTableFromCanvas = (tableName: string) => {
    // Remove related edges
    setEdges((prevEdges) =>
      prevEdges.filter(
        edge => edge.source !== tableName && edge.target !== tableName
      )
    );

    // Remove the node
    setNodes((prevNodes) =>
      prevNodes.filter(node => node.id !== tableName)
    );

    // Update selected tables
    setSelectedTables((prevTables) =>
      prevTables.filter(table => table !== tableName)
    );
  };

  // Toggle table expansion in the drawer
  const toggleTableExpansion = (tableName: string) => {
    setExpandedTable(expandedTable === tableName ? null : tableName);
  };

  // Save the current data model
  const saveDataModel = async () => {
    try {
      if (!selectedDataSource) {
        throw new Error('No data source selected');
      }

      setLoading(true);

      // Extract relationships from edges
      const relationships: Relationship[] = edges.map(edge => ({
        id: edge.id,
        sourceTable: edge.source,
        sourceColumn: edge.data?.sourceColumn,
        targetTable: edge.target,
        targetColumn: edge.data?.targetColumn,
        type: edge.data?.type || 'one-to-many'
      }));

      // Create layout data for saving node positions
      const layoutData = {
        nodes: nodes.map(node => ({
          id: node.id,
          position: node.position,
          data: {
            label: node.data?.label || node.id
          }
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          data: edge.data
        })),
        viewport: { x: 0, y: 0, zoom: 1 } // Default viewport
      };

      // Prepare data model to save
      const modelToSave = {
        id: dataModel?.id || '',
        name: dataModelName,
        dataSourceId: selectedDataSource.id,
        tables: selectedTables,
        relationships,
        layout: JSON.stringify(layoutData) // Store layout as JSON string
      };

      // Save through API
      let savedModel;
      if (dataModel?.id) {
        savedModel = await dataObjectsApi.updateDataModel(dataModel.id, modelToSave);
      } else {
        savedModel = await dataObjectsApi.createDataModel(modelToSave);
      }

      setDataModel(savedModel);

      // Show success notification
      setNotification({
        open: true,
        message: 'Data model saved successfully',
        severity: 'success'
      });

      setLoading(false);
      setOpenSaveModelDialog(false);

      // Refresh saved models list
      fetchSavedModels();
    } catch (err) {
      console.error('Error saving data model:', err);
      setNotification({
        open: true,
        message: `Failed to save data model: ${err instanceof Error ? err.message : 'Unknown error'}`,
        severity: 'error'
      });
      setLoading(false);
    }
  };

  // Load a data model
  const loadDataModel = async (modelId: string) => {
    try {
      setLoading(true);

      // Get the model from API
      const model = await dataObjectsApi.getDataModel(modelId);
      setDataModel(model);
      setDataModelName(model.name);

      // Clear existing canvas
      setNodes([]);
      setEdges([]);
      setSelectedTables([]);

      // Parse layout information if available
      let layoutData: any = null;
      if (model.layout) {
        try {
          layoutData = JSON.parse(model.layout);
          console.log('Loaded layout data:', layoutData);
        } catch (parseErr) {
          console.error('Error parsing layout data:', parseErr);
        }
      }

      // Add tables from the model with positions from layout if available
      const newNodes: Node[] = [];
      for (const tableName of model.tables) {
        const table = tables.find(t => t.name === tableName);
        if (table) {
          // Find position from layout if available
          let position = { x: 250 + (newNodes.length % 3) * 300, y: 100 + Math.floor(newNodes.length / 3) * 400 };
          if (layoutData?.nodes) {
            const layoutNode = layoutData.nodes.find((n: any) => n.id === tableName);
            if (layoutNode?.position) {
              position = layoutNode.position;
            }
          }

          // Create node
          const newNode: Node = {
            id: table.name,
            type: 'schemaTable',
            position,
            data: {
              label: table.name,
              columns: table.columns
            }
          };

          newNodes.push(newNode);
          setSelectedTables((prevTables) => [...prevTables, table.name]);
        }
      }

      // Set all nodes at once
      setNodes(newNodes);

      // Add relationships as edges
      const newEdges = model.relationships.map(rel => ({
        id: rel.id,
        source: rel.sourceTable,
        target: rel.targetTable,
        sourceHandle: `${rel.sourceTable}__${rel.sourceColumn}`,
        targetHandle: `${rel.targetTable}__${rel.targetColumn}`,
        type: 'relationship',
        animated: true,
        data: {
          type: rel.type,
          sourceColumn: rel.sourceColumn,
          targetColumn: rel.targetColumn
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20
        }
      }));

      setEdges(newEdges);

      setLoading(false);
      setOpenLoadModelDialog(false);

      // Show success notification
      setNotification({
        open: true,
        message: 'Data model loaded successfully',
        severity: 'success'
      });
    } catch (err) {
      console.error('Error loading data model:', err);
      setError(`Failed to load data model: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLoading(false);
    }
  };

  // Filter tables based on search query
  const filteredTables = tables.filter(table =>
    table.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* Left Drawer for Tables */}
      <Drawer
        variant="persistent"
        anchor="left"
        open={showTableDrawer}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            position: 'relative',
            height: '100%',
            border: 'none'
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Database Tables
          </Typography>

          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search tables..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
              size: 'small'
            }}
            size="small"
            sx={{ mb: 2 }}
          />

          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={40} />
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <List
            sx={{
              maxHeight: 'calc(100vh - 220px)',
              overflow: 'auto',
              bgcolor: 'background.paper'
            }}
          >
            {filteredTables.map((table) => (
              <React.Fragment key={table.name}>
                <ListItem
                  disablePadding
                  secondaryAction={
                    selectedTables.includes(table.name) ? (
                      <IconButton
                        edge="end"
                        onClick={() => removeTableFromCanvas(table.name)}
                        size="small"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    ) : (
                      <IconButton
                        edge="end"
                        onClick={() => addTableToCanvas(table)}
                        size="small"
                      >
                        <AddIcon fontSize="small" />
                      </IconButton>
                    )
                  }
                >
                  <ListItemButton onClick={() => toggleTableExpansion(table.name)}>
                    <ListItemIcon>
                      <TableViewIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={table.name}
                      secondary={`${table.columns.length} columns`}
                    />
                    {expandedTable === table.name ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </ListItemButton>
                </ListItem>

                <Collapse in={expandedTable === table.name} timeout="auto" unmountOnExit>
                  <List component="div" disablePadding>
                    {table.columns.map((column) => (
                      <ListItem key={`${table.name}-${column.name}`} sx={{ pl: 4 }}>
                        <ListItemText
                          primary={column.name}
                          secondary={column.type}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                        {column.isPrimaryKey && (
                          <Chip label="PK" size="small" color="primary" variant="outlined" sx={{ ml: 1 }} />
                        )}
                        {column.isForeignKey && (
                          <Chip label="FK" size="small" color="secondary" variant="outlined" sx={{ ml: 1 }} />
                        )}
                      </ListItem>
                    ))}
                  </List>
                </Collapse>
                <Divider />
              </React.Fragment>
            ))}

            {filteredTables.length === 0 && !loading && (
              <ListItem>
                <ListItemText
                  primary="No tables found"
                  secondary={searchQuery ? "Try a different search query" : "Connect to a database with tables"}
                />
              </ListItem>
            )}
          </List>
        </Box>
      </Drawer>

      {/* Main Content - ReactFlow Canvas */}
      <Box
        sx={{
          flexGrow: 1,
          height: '100%',
          position: 'relative',
          bgcolor: '#f5f5f5'
        }}
      >
        {/* Toolbar */}
        <AppBar
          position="relative"
          color="default"
          elevation={0}
          sx={{ borderBottom: '1px solid rgba(0, 0, 0, 0.12)' }}
        >
          <Toolbar variant="dense">
            <IconButton
              edge="start"
              color="inherit"
              onClick={() => setShowTableDrawer(!showTableDrawer)}
            >
              {showTableDrawer ? <ArrowBackIcon /> : <TableViewIcon />}
            </IconButton>

            <Typography variant="h6" sx={{ ml: 2, flexGrow: 1 }}>
              {dataModel ? dataModel.name : 'Data Model Canvas'}
            </Typography>

            <Button
              startIcon={<LinkIcon />}
              onClick={() => setOpenRelationshipDialog(true)}
              disabled={selectedTables.length < 2}
              sx={{ mr: 1 }}
            >
              Add Relationship
            </Button>

            <Button
              startIcon={<SaveIcon />}
              variant="contained"
              color="primary"
              onClick={() => setOpenSaveModelDialog(true)}
              disabled={selectedTables.length === 0}
            >
              Save Model
            </Button>
          </Toolbar>
        </AppBar>

        {/* ReactFlow Canvas */}
        <div ref={reactFlowWrapper} style={{ width: '100%', height: 'calc(100% - 48px)' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background />
          </ReactFlow>
        </div>
      </Box>

      {/* Relationship Dialog */}
      <Dialog open={openRelationshipDialog} onClose={() => setOpenRelationshipDialog(false)}>
        <DialogTitle>Create Relationship</DialogTitle>
        <DialogContent>
          <DialogContentText gutterBottom>
            Define the relationship between these tables:
          </DialogContentText>

          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Source Table</Typography>
              <Typography variant="body1">{relationshipForm.sourceTable}</Typography>
              <Typography variant="caption" color="text.secondary">
                Column: {relationshipForm.sourceColumn}
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Target Table</Typography>
              <Typography variant="body1">{relationshipForm.targetTable}</Typography>
              <Typography variant="caption" color="text.secondary">
                Column: {relationshipForm.targetColumn}
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <TextField
                select
                fullWidth
                label="Relationship Type"
                value={relationshipForm.type}
                onChange={(e) => setRelationshipForm({
                  ...relationshipForm,
                  type: e.target.value as 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many'
                })}
                SelectProps={{
                  native: true,
                }}
                variant="outlined"
                margin="normal"
              >
                <option value="one-to-one">One-to-One</option>
                <option value="one-to-many">One-to-Many</option>
                <option value="many-to-one">Many-to-One</option>
                <option value="many-to-many">Many-to-Many</option>
              </TextField>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenRelationshipDialog(false)}>Cancel</Button>
          <Button onClick={createRelationship} variant="contained" color="primary">Create</Button>
        </DialogActions>
      </Dialog>

      {/* Save Model Dialog */}
      <Dialog open={openSaveModelDialog} onClose={() => setOpenSaveModelDialog(false)}>
        <DialogTitle>Save Data Model</DialogTitle>
        <DialogContent>
          <DialogContentText gutterBottom>
            Provide a name for your data model:
          </DialogContentText>

          <TextField
            autoFocus
            margin="dense"
            label="Model Name"
            type="text"
            fullWidth
            variant="outlined"
            value={dataModelName}
            onChange={(e) => setDataModelName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenSaveModelDialog(false)}>Cancel</Button>
          <Button
            onClick={saveDataModel}
            variant="contained"
            color="primary"
            disabled={!dataModelName.trim()}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setNotification({ ...notification, open: false })}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataObjectsCanvas;
