import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Alert,
  AppBar,
  Toolbar,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  TextField,
  Snackbar
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Save as SaveIcon,
  ArrowBack as ArrowBackIcon
} from '@mui/icons-material';
import { Link, useSearchParams } from 'react-router-dom';
import useDataSourceStore from '../../store/dataSourceStore';
import { DataCanvas } from '../../components/reporting/data-canvas';
import * as dataObjectsApi from '../../components/data-object-canvas/data-objects-api';
import { Node, Edge } from 'reactflow';
import { DataModel, Relationship, TableSchema } from '../../types/reporting';

// Extended DataModel interface with layout information
interface ExtendedDataModel extends DataModel {
  layout?: string; // JSON string containing layout information
}

// Define the structure of the canvas data for saving
interface CanvasData {
  nodes: {
    id: string;
    position: {
      x: number;
      y: number;
    };
    data: {
      label: string;
      columns: any[];
    };
    type: string;
  }[];
  edges: {
    id: string;
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
    data: {
      type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
      sourceColumn: string;
      targetColumn: string;
    };
    type: string;
  }[];
  viewport: {
    x: number;
    y: number;
    zoom: number;
  };
};

const DataCanvasPage: React.FC = () => {
  // Extract URL parameters
  const [searchParams] = useSearchParams();
  const modelId = searchParams.get('modelId');
  const isEditMode = searchParams.get('edit') === 'true'; // Will be used for edit mode functionality

  // Data source state
  const {
    dataSources,
    selectedDataSource,
    selectDataSource,
    fetchDataSources,
    loading: dataSourceLoading
  } = useDataSourceStore();

  // Canvas state
  const [tables, setTables] = useState<TableSchema[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data model state
  const [dataModel, setDataModel] = useState<DataModel | null>(null);
  const [dataModelName, setDataModelName] = useState('New Data Model');

  // UI state
  const [openSaveDialog, setOpenSaveDialog] = useState(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Fetch data sources on component mount
  useEffect(() => {
    if (dataSources.length === 0) {
      fetchDataSources();
    }
  }, [dataSources.length, fetchDataSources]);

  // Fetch tables when data source changes
  useEffect(() => {
    if (selectedDataSource) {
      fetchTableSchemas(selectedDataSource.id);
    } else if (dataSources.length > 0) {
      selectDataSource(dataSources[0].id);
    }
  }, [selectedDataSource, dataSources, selectDataSource]);

  // Load data model from URL parameter if provided
  useEffect(() => {
    const loadModelFromUrl = async () => {
      if (modelId && selectedDataSource) {
        try {
          setLoading(true);

          // Fetch the data model
          const model = await dataObjectsApi.getDataModel(modelId) as ExtendedDataModel;
          console.log('Loaded model:', model);
          setDataModel(model);
          setDataModelName(model.name);

          // Make sure we have the correct data source selected
          if (model.dataSourceId !== selectedDataSource.id) {
            selectDataSource(model.dataSourceId);
            return; // The data source change will trigger a reload of tables
          }

          // Wait for tables to be loaded
          if (tables.length === 0) {
            const tableSchemas = await dataObjectsApi.getDatabaseSchema(model.dataSourceId);
            setTables(tableSchemas);
          }

          // Clear existing canvas
          setNodes([]);
          setEdges([]);

          // Add tables from the model
          const newNodes: Node[] = [];

          // Check if we have layout information
          let layout: any;
          try {
            if (model.layout) {
              layout = JSON.parse(model.layout);
              console.log('Parsed layout:', layout);
            }
          } catch (err) {
            console.error('Error parsing layout:', err);
          }

          // Add tables to canvas with positions from layout if available
          for (const tableName of model.tables) {
            const table = tables.find(t => t.name === tableName);
            if (table) {
              console.log(`Found table ${tableName} in available tables:`, table);
              // Find position from layout if available
              let position = { x: 100 + newNodes.length * 200, y: 100 };
              if (layout?.nodes) {
                const layoutNode = layout.nodes.find((n: any) => n.id === tableName);
                if (layoutNode?.position) {
                  position = layoutNode.position;
                  console.log(`Found position for ${tableName}:`, position);
                }
              }

              // Create node
              const newNode: Node = {
                id: table.name,
                type: 'tableNode',
                position,
                data: {
                  label: table.name,
                  columns: table.columns
                }
              };

              newNodes.push(newNode);
              console.log(`Added node for table ${table.name}:`, newNode);
            } else {
              console.warn(`Table ${tableName} not found in available tables`);
            }
          }

          // Add all nodes at once
          console.log('Setting nodes:', newNodes);
          // Use a timeout to ensure state updates properly
          setTimeout(() => {
            setNodes(newNodes);
          }, 0);

          // Add relationships as edges
          if (model.relationships && model.relationships.length > 0) {
            console.log('Processing relationships:', model.relationships);
            const newEdges: Edge[] = model.relationships.map(rel => {
              console.log(`Creating edge from ${rel.sourceTable} to ${rel.targetTable}`);
              return {
                id: rel.id,
                source: rel.sourceTable,
                target: rel.targetTable,
                sourceHandle: `${rel.sourceTable}__${rel.sourceColumn}`,
                targetHandle: `${rel.targetTable}__${rel.targetColumn}`,
                data: {
                  sourceColumn: rel.sourceColumn,
                  targetColumn: rel.targetColumn,
                  type: rel.type || 'one-to-many'
                },
                type: 'relationship'
              };
            });

            console.log('Setting edges:', newEdges);
            // Use a timeout to ensure state updates properly after nodes are set
            setTimeout(() => {
              setEdges(newEdges);
            }, 100);
          }

          setLoading(false);
          setNotification({
            open: true,
            message: `Data model "${model.name}" loaded successfully`,
            severity: 'success'
          });
        } catch (err) {
          console.error('Error loading data model:', err);
          setError(`Failed to load data model: ${err instanceof Error ? err.message : 'Unknown error'}`);
          setNotification({
            open: true,
            message: `Error loading data model: ${err instanceof Error ? err.message : 'Unknown error'}`,
            severity: 'error'
          });
          setLoading(false);
        }
      }
    };

    loadModelFromUrl();
  }, [modelId, selectedDataSource, tables, selectDataSource]);

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

  // Handle save dialog open
  const handleOpenSaveDialog = (canvasData?: CanvasData) => {
    if (canvasData) {
      // Store the canvas data for saving
      setNodes(canvasData.nodes as Node[]);
      setEdges(canvasData.edges as Edge[]);
    }
    setOpenSaveDialog(true);
  };

  // Handle save dialog close
  const handleCloseSaveDialog = () => {
    setOpenSaveDialog(false);
  };

  // Save the data model
  const saveDataModel = async () => {
    try {
      if (!selectedDataSource) {
        throw new Error('No data source selected');
      }

      setLoading(true);

      // Extract tables and their positions from nodes
      const tableNames = nodes.map(node => node.id);

      // Store the canvas layout information
      const canvasLayout = {
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
        viewport: { x: 0, y: 0, zoom: 1 } // Default viewport if not available
      };

      // Extract relationships from edges
      const relationships: Relationship[] = edges.map(edge => ({
        id: edge.id,
        sourceTable: edge.source,
        sourceColumn: edge.data?.sourceColumn || '',
        targetTable: edge.target,
        targetColumn: edge.data?.targetColumn || '',
        type: edge.data?.type || 'one-to-many'
      }));

      // Prepare data model to save
      const modelToSave: Partial<ExtendedDataModel> = {
        id: dataModel?.id || '',
        name: dataModelName,
        dataSourceId: selectedDataSource.id,
        tables: tableNames,
        relationships,
        layout: JSON.stringify(canvasLayout) // Store the canvas layout as a JSON string
      };

      // Save the model
      let savedModel: DataModel;
      if (dataModel?.id) {
        savedModel = await dataObjectsApi.updateDataModel(dataModel.id, modelToSave);
      } else {
        savedModel = await dataObjectsApi.createDataModel(modelToSave);
      }

      setDataModel(savedModel);
      setNotification({
        open: true,
        message: `Data model "${savedModel.name}" saved successfully`,
        severity: 'success'
      });

      setLoading(false);
      setOpenSaveDialog(false);
    } catch (err) {
      console.error('Error saving data model:', err);
      setNotification({
        open: true,
        message: `Error saving data model: ${err instanceof Error ? err.message : 'Unknown error'}`,
        severity: 'error'
      });
      setLoading(false);
    }
  };

  // Handle notification close
  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  return (
    <Box sx={{ height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <AppBar position="static" color="default" elevation={0}>
        <Toolbar>
          <IconButton edge="start" component={Link} to="/reporting">
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h6" sx={{ ml: 2, flexGrow: 1 }}>
            Data Canvas {dataModel && `- ${dataModel.name}`}
          </Typography>

          {loading && <CircularProgress size={24} sx={{ mr: 2 }} />}

          <Button
            startIcon={<RefreshIcon />}
            onClick={() => selectedDataSource && fetchTableSchemas(selectedDataSource.id)}
            disabled={loading || !selectedDataSource}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>

          <Button
            startIcon={<SaveIcon />}
            variant="contained"
            color="primary"
            onClick={() => handleOpenSaveDialog()}
            disabled={loading || nodes.length === 0}
          >
            Save Model
          </Button>
        </Toolbar>
      </AppBar>

      {/* Error Alert */}
      {error && (
        <Alert
          severity="error"
          onClose={() => setError(null)}
          sx={{ borderRadius: 0 }}
        >
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Box sx={{ flexGrow: 1, position: 'relative', display: 'flex' }}>
        {dataSourceLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : dataSources.length === 0 ? (
          <Paper sx={{ p: 3, m: 2, textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              No Data Sources Available
            </Typography>
            <Typography variant="body1" paragraph>
              You need to create a data source before you can use the Data Canvas.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              component={Link}
              to="/reporting/datasources/create"
            >
              Create Data Source
            </Button>
          </Paper>
        ) : (
          <DataCanvas
            tables={tables}
            initialNodes={nodes}
            initialEdges={edges}
            onNodesChange={(updatedNodes) => {
              console.log('Nodes updated:', updatedNodes);
              setNodes(updatedNodes);
            }}
            onEdgesChange={(updatedEdges) => {
              console.log('Edges updated:', updatedEdges);
              setEdges(updatedEdges);
            }}
            onSave={handleOpenSaveDialog}
          />
        )}
      </Box>

      {/* Save Dialog */}
      <Dialog open={openSaveDialog} onClose={handleCloseSaveDialog}>
        <DialogTitle>Save Data Model</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Enter a name for your data model. This will save the current state of your canvas.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Data Model Name"
            fullWidth
            value={dataModelName}
            onChange={(e) => setDataModelName(e.target.value)}
            variant="outlined"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSaveDialog}>Cancel</Button>
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

      {/* Notification */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseNotification}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataCanvasPage;
