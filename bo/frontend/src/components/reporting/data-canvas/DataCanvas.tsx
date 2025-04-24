import React, { useState, useCallback, useRef, useEffect } from 'react';
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
  ReactFlowProvider,
  Panel,
  ConnectionLineType,
  ReactFlowInstance,
  ConnectionMode
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Box } from '@mui/material';

// Import custom components
import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import { NodePalette, CanvasControls } from './controls';
import { TableSchema, ColumnSchema } from '../../../types/reporting';

// Styles for the data canvas
const canvasStyles = {
  width: '100%',
  height: '100%',
  background: '#f8f8f8',
  borderRadius: 1,
  overflow: 'hidden'
};

// Default viewport
const defaultViewport = { x: 0, y: 0, zoom: 1 };

export interface DataCanvasProps {
  tables: TableSchema[];
  initialNodes?: Node[];
  initialEdges?: Edge[];
  onNodesChange?: (nodes: Node[]) => void;
  onEdgesChange?: (edges: Edge[]) => void;
  onSave?: (canvasData: CanvasData) => void;
  readOnly?: boolean;
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
      columns: ColumnSchema[];
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
}

/**
 * DataCanvas component
 * Main component for the data canvas
 * Integrates ReactFlow with custom nodes, edges, and controls
 */
const DataCanvas: React.FC<DataCanvasProps> = ({
  tables = [],
  initialNodes = [],
  initialEdges = [],
  onNodesChange,
  onEdgesChange,
  onSave,
  readOnly = false
}) => {
  // const theme = useTheme(); // Uncomment if needed for theming
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [nodes, setNodes, onNodesChangeInternal] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChangeInternal] = useEdgesState(initialEdges);

  // Log initial nodes and edges for debugging and update internal state when props change
  useEffect(() => {
    console.log('DataCanvas initialNodes:', initialNodes);
    console.log('DataCanvas initialEdges:', initialEdges);

    // Update internal state when initialNodes or initialEdges change from props
    if (initialNodes.length > 0) {
      setNodes(initialNodes);
    }

    if (initialEdges.length > 0) {
      setEdges(initialEdges);
    }
  }, [initialNodes, initialEdges]);
  const [showGrid, setShowGrid] = useState(true);
  const [selectedElements, setSelectedElements] = useState<{ nodes: Node[]; edges: Edge[] }>({
    nodes: [],
    edges: []
  });

  // Handle node changes
  const handleNodesChange = useCallback(
    (changes: any) => {
      onNodesChangeInternal(changes);
      // Use setTimeout to ensure we get the latest state after the internal state update
      setTimeout(() => {
        if (onNodesChange) {
          onNodesChange(nodes);
        }
      }, 0);
    },
    [nodes, onNodesChange, onNodesChangeInternal]
  );

  // Handle edge changes
  const handleEdgesChange = useCallback(
    (changes: any) => {
      onEdgesChangeInternal(changes);
      // Use setTimeout to ensure we get the latest state after the internal state update
      setTimeout(() => {
        if (onEdgesChange) {
          onEdgesChange(edges);
        }
      }, 0);
    },
    [edges, onEdgesChange, onEdgesChangeInternal]
  );

  // Handle connection between nodes
  const onConnect = useCallback(
    (connection: Connection) => {
      // Extract source and target handle information
      const sourceHandle = connection.sourceHandle || '';
      const targetHandle = connection.targetHandle || '';

      // Determine if this is a main connection or a column-specific connection
      const isMainConnection = sourceHandle.includes('main') || targetHandle.includes('main');

      // For main connections, use default column names
      let sourceColumn = '';
      let targetColumn = '';

      if (isMainConnection) {
        // For main connections, we'll use a generic relationship
        sourceColumn = 'main';
        targetColumn = 'main';
      } else {
        // For column-specific connections, extract the column names
        sourceColumn = sourceHandle.split('__')[1] || '';
        targetColumn = targetHandle.split('__')[1] || '';
      }

      // Create the new edge
      const newEdge = {
        ...connection,
        type: 'relationship',
        animated: true,
        data: {
          type: 'one-to-many', // Default relationship type
          sourceColumn,
          targetColumn
        }
      };

      const updatedEdges = addEdge(newEdge, edges);
      setEdges(updatedEdges);

      if (onEdgesChange) {
        onEdgesChange(updatedEdges);
      }
    },
    [edges, onEdgesChange, setEdges]
  );

  // Handle edge update (e.g., changing relationship type)
  const handleEdgeUpdate = useCallback(
    (edgeId: string, newData: any) => {
      setEdges((eds) =>
        eds.map((edge) => {
          if (edge.id === edgeId) {
            return {
              ...edge,
              data: {
                ...edge.data,
                ...newData
              }
            };
          }
          return edge;
        })
      );

      if (onEdgesChange) {
        const updatedEdges = edges.map((edge) => {
          if (edge.id === edgeId) {
            return {
              ...edge,
              data: {
                ...edge.data,
                ...newData
              }
            };
          }
          return edge;
        });
        onEdgesChange(updatedEdges);
      }
    },
    [edges, onEdgesChange, setEdges]
  );

  // Handle node update (e.g., changing table name)
  const handleNodeUpdate = useCallback(
    (nodeId: string, newData: any) => {
      // If we're renaming a node, we need to update its ID as well
      if (newData.label && newData.label !== nodeId) {
        const newNodeId = newData.label;

        // Update the node with the new ID and data
        setNodes((nds) => {
          const updatedNodes = nds.map((node) => {
            if (node.id === nodeId) {
              return {
                ...node,
                id: newNodeId,
                data: {
                  ...node.data,
                  label: newData.label
                }
              };
            }
            return node;
          });

          // Also update any edges that reference this node
          setEdges((eds) => {
            return eds.map((edge) => {
              if (edge.source === nodeId) {
                return {
                  ...edge,
                  source: newNodeId
                };
              }
              if (edge.target === nodeId) {
                return {
                  ...edge,
                  target: newNodeId
                };
              }
              return edge;
            });
          });

          return updatedNodes;
        });

        // Notify parent component of node changes
        if (onNodesChange) {
          const updatedNodes = nodes.map((node) => {
            if (node.id === nodeId) {
              return {
                ...node,
                id: newNodeId,
                data: {
                  ...node.data,
                  label: newData.label
                }
              };
            }
            return node;
          });
          onNodesChange(updatedNodes);
        }

        // Notify parent component of edge changes
        if (onEdgesChange) {
          const updatedEdges = edges.map((edge) => {
            if (edge.source === nodeId) {
              return {
                ...edge,
                source: newNodeId
              };
            }
            if (edge.target === nodeId) {
              return {
                ...edge,
                target: newNodeId
              };
            }
            return edge;
          });
          onEdgesChange(updatedEdges);
        }
      } else {
        // Regular update without ID change
        setNodes((nds) =>
          nds.map((node) => {
            if (node.id === nodeId) {
              return {
                ...node,
                data: {
                  ...node.data,
                  ...newData
                }
              };
            }
            return node;
          })
        );

        if (onNodesChange) {
          const updatedNodes = nodes.map((node) => {
            if (node.id === nodeId) {
              return {
                ...node,
                data: {
                  ...node.data,
                  ...newData
                }
              };
            }
            return node;
          });
          onNodesChange(updatedNodes);
        }
      }
    },
    [nodes, edges, onNodesChange, onEdgesChange, setNodes, setEdges]
  );

  // Handle renaming a table in the palette
  const handleRenameTable = useCallback(
    (oldName: string, newName: string) => {
      // Find if this table is already on the canvas
      const existingNode = nodes.find(node => node.id === oldName);

      if (existingNode) {
        // Update the node with the new name
        handleNodeUpdate(oldName, { label: newName });
      }
    },
    [nodes, handleNodeUpdate]
  );

  // Handle drag over for node creation
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop for table node creation
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      // Get the table data from the drag event
      const tableData = event.dataTransfer.getData('application/reactflow/table');
      if (!tableData) return;

      const table = JSON.parse(tableData) as TableSchema;

      // Get the position of the drop
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY
      });

      // Create a new node for the table
      const newNode: Node = {
        id: table.name,
        type: 'tableNode',
        position,
        data: {
          label: table.name,
          columns: table.columns
        }
      };

      // Add the node to the canvas
      setNodes(nodes => [...nodes, newNode]);

      // Notify parent component of node changes
      if (onNodesChange) {
        const updatedNodes = [...nodes, newNode];
        onNodesChange(updatedNodes);
      }
    },
    [reactFlowInstance, nodes, onNodesChange, setNodes]
  );

  // Handle selection change
  const onSelectionChange = useCallback(
    ({ nodes, edges }: { nodes: Node[]; edges: Edge[] }) => {
      setSelectedElements({ nodes, edges });
    },
    []
  );

  // Handle zoom in
  const handleZoomIn = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.zoomIn();
    }
  }, [reactFlowInstance]);

  // Handle zoom out
  const handleZoomOut = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.zoomOut();
    }
  }, [reactFlowInstance]);

  // Handle fit view
  const handleFitView = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.fitView();
    }
  }, [reactFlowInstance]);

  // Handle toggle grid
  const handleToggleGrid = useCallback(() => {
    setShowGrid(prev => !prev);
  }, []);

  // Handle delete selected elements
  const handleDeleteSelected = useCallback(() => {
    if (selectedElements.nodes.length > 0 || selectedElements.edges.length > 0) {
      // Remove selected nodes
      if (selectedElements.nodes.length > 0) {
        const nodeIdsToRemove = selectedElements.nodes.map(node => node.id);
        setNodes(nodes => nodes.filter(node => !nodeIdsToRemove.includes(node.id)));
      }

      // Remove selected edges
      if (selectedElements.edges.length > 0) {
        const edgeIdsToRemove = selectedElements.edges.map(edge => edge.id);
        setEdges(edges => edges.filter(edge => !edgeIdsToRemove.includes(edge.id)));
      }

      // Notify parent components
      if (onNodesChange && selectedElements.nodes.length > 0) {
        onNodesChange(nodes.filter(node => !selectedElements.nodes.map(n => n.id).includes(node.id)));
      }

      if (onEdgesChange && selectedElements.edges.length > 0) {
        onEdgesChange(edges.filter(edge => !selectedElements.edges.map(e => e.id).includes(edge.id)));
      }
    }
  }, [selectedElements, nodes, edges, onNodesChange, onEdgesChange, setNodes, setEdges]);

  // Create a JSON representation of the canvas for saving
  const createCanvasJSON = useCallback((): CanvasData => {
    // Create a data model object
    const dataModel = {
      nodes: nodes.map(node => ({
        id: node.id,
        position: {
          x: node.position.x,
          y: node.position.y
        },
        data: {
          label: node.data.label,
          columns: node.data.columns
        },
        type: node.type || 'tableNode'
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
        data: {
          type: edge.data?.type || 'one-to-many',
          sourceColumn: edge.data?.sourceColumn || '',
          targetColumn: edge.data?.targetColumn || ''
        },
        type: edge.type || 'relationship'
      })),
      viewport: reactFlowInstance ? reactFlowInstance.getViewport() : { x: 0, y: 0, zoom: 1 }
    };

    return dataModel;
  }, [nodes, edges, reactFlowInstance]);

  // Handle save
  const handleSave = useCallback(() => {
    if (onSave) {
      const canvasData = createCanvasJSON();
      onSave(canvasData);
    }
  }, [onSave, createCanvasJSON]);

  // Debug current state
  useEffect(() => {
    console.log('Current nodes state:', nodes);
    console.log('Current edges state:', edges);
  }, [nodes, edges]);

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <ReactFlowProvider>
        <Box ref={reactFlowWrapper} sx={canvasStyles}>
          <ReactFlow
            nodes={nodes.map(node => ({
              ...node,
              data: {
                ...node.data,
                onNodeUpdate: handleNodeUpdate
              }
            }))}
            edges={edges.map(edge => ({
              ...edge,
              data: {
                ...edge.data,
                updateEdge: handleEdgeUpdate
              }
            }))}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onSelectionChange={onSelectionChange}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultViewport={defaultViewport}
            minZoom={0.1}
            maxZoom={4}
            fitView={nodes.length > 0}
            attributionPosition="bottom-left"
            connectionLineType={ConnectionLineType.Bezier}
            connectionLineStyle={{ stroke: '#999', strokeWidth: 2 }}
            connectionMode={ConnectionMode.Loose}
            deleteKeyCode={['Backspace', 'Delete']}
            multiSelectionKeyCode={['Control', 'Meta']}
            snapToGrid={true}
            snapGrid={[15, 15]}
            selectNodesOnDrag={false}
            panOnDrag={!readOnly}
            zoomOnScroll={!readOnly}
            panOnScroll={false}
            elementsSelectable={!readOnly}
            nodesConnectable={!readOnly}
            nodesDraggable={!readOnly}
          >
            {/* Background */}
            {showGrid && <Background />}

            {/* Controls */}
            <Controls showInteractive={!readOnly} />

            {/* Node Palette */}
            <Panel position="top-left">
              <NodePalette
                tables={tables}
                readOnly={readOnly}
                onRenameTable={handleRenameTable}
              />
            </Panel>

            {/* Canvas Controls */}
            <Panel position="top-right">
              <CanvasControls
                onZoomIn={handleZoomIn}
                onZoomOut={handleZoomOut}
                onFitView={handleFitView}
                onToggleGrid={handleToggleGrid}
                onDelete={!readOnly ? handleDeleteSelected : undefined}
                onSave={!readOnly && onSave ? handleSave : undefined}
                showGrid={showGrid}
                hasSelection={selectedElements.nodes.length > 0 || selectedElements.edges.length > 0}
              />
            </Panel>

            {/* Mini Map */}
            <MiniMap
              nodeStrokeWidth={3}
              zoomable
              pannable
              nodeColor={() => '#1976d2'} // Primary color for all table nodes
            />
          </ReactFlow>
        </Box>
      </ReactFlowProvider>
    </Box>
  );
};

export default DataCanvas;
