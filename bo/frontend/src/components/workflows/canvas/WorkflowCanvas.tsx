import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  ReactFlowInstance,
  Panel,
  ConnectionLineType,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Box, Paper, Typography, useTheme, Dialog } from '@mui/material';

// Import custom node and edge components
import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import { NodePalette, CanvasControls } from './controls';
import {
  AgentNodeForm,
  ToolNodeForm,
  ConditionNodeForm,
  InputNodeForm,
  OutputNodeForm,
  GroupNodeForm,
  ConsensusNodeForm,
  TemporalCoordinationNodeForm,
  DatabaseNodeForm,
  FilesystemNodeForm,
  APINodeForm
} from './forms';

// Import types
import { NodeData, NodeType, createNode, DEFAULT_NODE_PORTS, NODE_TYPE_CONFIG } from './types/nodeTypes';
import { EdgeData, EdgeType, EDGE_TYPE_CONFIG } from './types/edgeTypes';

// Styles for the workflow canvas
const canvasStyles = {
  width: '100%',
  height: '100%',
  background: '#f8f8f8',
  borderRadius: 1,
  overflow: 'hidden'
};

// Default viewport
const defaultViewport = { x: 0, y: 0, zoom: 1 };

export interface WorkflowCanvasProps {
  initialNodes?: Node<NodeData>[];
  initialEdges?: Edge<EdgeData>[];
  onNodesChange?: (nodes: Node<NodeData>[]) => void;
  onEdgesChange?: (edges: Edge<EdgeData>[]) => void;
  onSave?: (nodes: Node<NodeData>[], edges: Edge<EdgeData>[]) => void;
  readOnly?: boolean;
}

/**
 * WorkflowCanvas component
 * Main component for the workflow canvas
 * Integrates ReactFlow with custom nodes, edges, and controls
 */
const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  initialNodes = [],
  initialEdges = [],
  onNodesChange,
  onEdgesChange,
  onSave,
  readOnly = false
}) => {
  const theme = useTheme();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [nodes, setNodes, onNodesChangeInternal] = useNodesState<NodeData>(initialNodes);
  const [edges, setEdges, onEdgesChangeInternal] = useEdgesState<EdgeData>(initialEdges);
  const [showGrid, setShowGrid] = useState(true);
  const [selectedElements, setSelectedElements] = useState<{ nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] }>({
    nodes: [],
    edges: []
  });
  const [editingNode, setEditingNode] = useState<Node<NodeData> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // ReactFlow instance will be used for viewport operations
  
  // Handle node edit
  const handleNodeEdit = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (node) {
      setEditingNode(node);
    }
  }, [nodes]);
  
  // Handle node edit save
  const handleNodeSave = useCallback((updatedNodeData: NodeData) => {
    if (!editingNode) return;
    
    setIsLoading(true);
    
    // Update the node with the new data
    const updatedNodes = nodes.map(node => {
      if (node.id === updatedNodeData.id) {
        return {
          ...node,
          data: updatedNodeData
        };
      }
      return node;
    });
    
    setNodes(updatedNodes);
    
    // Notify parent component of node changes
    if (onNodesChange) {
      onNodesChange(updatedNodes);
    }
    
    // Close the form
    setIsLoading(false);
    setEditingNode(null);
  }, [editingNode, nodes, setNodes, onNodesChange]);
  
  // Handle node edit cancel
  const handleNodeEditCancel = useCallback(() => {
    setEditingNode(null);
  }, []);
  
  // Render the appropriate form based on node type
  const renderNodeForm = useMemo(() => {
    if (!editingNode) return null;
    
    const nodeType = editingNode.type as NodeType;
    
    switch (nodeType) {
      case NodeType.AGENT:
        return (
          <AgentNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
            availableAgents={[]} // This would come from a prop or API call
          />
        );
      case NodeType.TOOL:
        return (
          <ToolNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
            availableTools={[]} // This would come from a prop or API call
          />
        );
      case NodeType.CONDITION:
        return (
          <ConditionNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.INPUT:
        return (
          <InputNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.OUTPUT:
        return (
          <OutputNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.GROUP:
        return (
          <GroupNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
            availableNodes={nodes}
          />
        );
      case NodeType.CONSENSUS:
        return (
          <ConsensusNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.TEMPORAL_COORDINATION:
        return (
          <TemporalCoordinationNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.DATABASE:
        return (
          <DatabaseNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.FILESYSTEM:
        return (
          <FilesystemNodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      case NodeType.API:
        return (
          <APINodeForm
            open={!!editingNode}
            node={editingNode.data}
            onClose={handleNodeEditCancel}
            onSave={handleNodeSave}
            isLoading={isLoading}
          />
        );
      default:
        return null;
    }
  }, [editingNode, isLoading, handleNodeEditCancel, handleNodeSave, nodes]);
  
  // Handle node selection
  const onSelectionChange = useCallback(
    ({ nodes, edges }: { nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] }) => {
      setSelectedElements({ nodes, edges });
    },
    []
  );
  
  // Handle connection between nodes
  const onConnect = useCallback(
    (connection: Connection) => {
      // Determine edge type based on the source and target handles
      const sourceNode = nodes.find(node => node.id === connection.source);
      const targetNode = nodes.find(node => node.id === connection.target);
      
      if (!sourceNode || !targetNode) return;
      
      // Default to data edge
      let edgeType = EdgeType.DATA;
      
      // Check if this is a control flow edge (e.g., from a condition node)
      if (sourceNode.type === NodeType.CONDITION) {
        edgeType = EdgeType.CONTROL;
      }
      
      // Create edge data
      const edgeData: EdgeData = {
        id: `edge-${Date.now()}`,
        source: connection.source || '',
        sourceHandle: connection.sourceHandle || '',
        target: connection.target || '',
        targetHandle: connection.targetHandle || '',
        type: edgeType,
        label: EDGE_TYPE_CONFIG[edgeType].label,
        animated: EDGE_TYPE_CONFIG[edgeType].animated,
        style: {}
      };
      
      // Add the edge with the appropriate type and marker
      const newEdge = {
        id: edgeData.id,
        source: connection.source || '',
        target: connection.target || '',
        sourceHandle: connection.sourceHandle || '',
        targetHandle: connection.targetHandle || '',
        type: edgeType,
        data: edgeData,
        animated: edgeData.animated,
        style: { stroke: EDGE_TYPE_CONFIG[edgeType].color },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_TYPE_CONFIG[edgeType].color
        }
      } as Edge<EdgeData>;
      
      setEdges(edges => addEdge(newEdge, edges));
      
      // Notify parent component of edge changes
      if (onEdgesChange) {
        onEdgesChange([...edges, newEdge]);
      }
    },
    [nodes, edges, onEdgesChange]
  );
  
  // Handle drag over for node creation
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);
  
  // Handle drop for node creation
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      
      if (!reactFlowWrapper.current || !reactFlowInstance) return;
      
      // Get the node type from the drag event
      const nodeType = event.dataTransfer.getData('application/reactflow/type') as NodeType;
      
      // Get the position of the drop
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY
      });
      
      // Create a new node
      const nodeData = createNode(nodeType, position);
      
      // Ensure the node has ports
      if (!nodeData.ports) {
        nodeData.ports = DEFAULT_NODE_PORTS[nodeType];
      }
      
      // Convert NodeData to Node<NodeData>
      const newNode: Node<NodeData> = {
        id: nodeData.id,
        type: nodeData.type,
        position: nodeData.position,
        data: nodeData,
      };
      
      // Add the node to the canvas
      setNodes(nodes => [...nodes, newNode]);
      
      // Notify parent component of node changes
      if (onNodesChange) {
        const updatedNodes = [...nodes, newNode];
        onNodesChange(updatedNodes);
      }
    },
    [reactFlowInstance, nodes, onNodesChange]
  );
  
  // Handle node drag start
  const onNodeDragStart = useCallback(
    (event: React.DragEvent<HTMLDivElement>, nodeType: NodeType) => {
      event.dataTransfer.setData('application/reactflow/type', nodeType);
      event.dataTransfer.effectAllowed = 'move';
    },
    []
  );
  
  // Handle zoom in
  const handleZoomIn = useCallback(() => {
    if (reactFlowInstance) {
      const { x, y, zoom } = reactFlowInstance.getViewport();
      reactFlowInstance.setViewport({ x, y, zoom: zoom * 1.2 });
    }
  }, [reactFlowInstance]);
  
  // Handle zoom out
  const handleZoomOut = useCallback(() => {
    if (reactFlowInstance) {
      const { x, y, zoom } = reactFlowInstance.getViewport();
      reactFlowInstance.setViewport({ x, y, zoom: zoom / 1.2 });
    }
  }, [reactFlowInstance]);
  
  // Handle fit view
  const handleFitView = useCallback(() => {
    if (reactFlowInstance) {
      reactFlowInstance.fitView({ padding: 0.2 });
    }
  }, [reactFlowInstance]);
  
  // Handle toggle grid
  const handleToggleGrid = useCallback(() => {
    setShowGrid(prev => !prev);
  }, []);
  
  // Handle delete selected elements
  const handleDeleteSelected = useCallback(() => {
    if (selectedElements.nodes.length === 0 && selectedElements.edges.length === 0) return;
    
    // Delete selected nodes
    if (selectedElements.nodes.length > 0) {
      const nodeIdsToDelete = selectedElements.nodes.map(node => node.id);
      setNodes(nodes => nodes.filter(node => !nodeIdsToDelete.includes(node.id)));
    }
    
    // Delete selected edges
    if (selectedElements.edges.length > 0) {
      const edgeIdsToDelete = selectedElements.edges.map(edge => edge.id);
      setEdges(edges => edges.filter(edge => !edgeIdsToDelete.includes(edge.id)));
    }
    
    // Notify parent component of changes
    if (onNodesChange) {
      onNodesChange(nodes.filter(node => !selectedElements.nodes.map(n => n.id).includes(node.id)));
    }
    
    if (onEdgesChange) {
      onEdgesChange(edges.filter(edge => !selectedElements.edges.map(e => e.id).includes(edge.id)));
    }
  }, [selectedElements, nodes, edges, setNodes, setEdges, onNodesChange, onEdgesChange]);
  
  // Handle save workflow
  const handleSave = useCallback(() => {
    if (onSave) {
      onSave(nodes, edges);
    }
  }, [nodes, edges, onSave]);
  
  // Update parent component when nodes or edges change
  useEffect(() => {
    if (onNodesChange) {
      onNodesChange(nodes);
    }
  }, [nodes, onNodesChange]);
  
  useEffect(() => {
    if (onEdgesChange) {
      onEdgesChange(edges);
    }
  }, [edges, onEdgesChange]);
  
  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      <ReactFlowProvider>
        <Box ref={reactFlowWrapper} sx={canvasStyles}>
          <ReactFlow
            nodes={nodes.map(node => ({
              ...node,
              data: {
                ...node.data,
                onNodeEdit: handleNodeEdit,
                onNodeDelete: (nodeId: string) => {
                  setNodes(nodes => nodes.filter(n => n.id !== nodeId));
                  if (onNodesChange) {
                    onNodesChange(nodes.filter(n => n.id !== nodeId));
                  }
                }
              }
            }))}
            edges={edges}
            onNodesChange={onNodesChangeInternal}
            onEdgesChange={onEdgesChangeInternal}
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
            fitView
            attributionPosition="bottom-left"
            connectionLineType={ConnectionLineType.Bezier}
            connectionLineStyle={{ stroke: '#999' }}
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
            edgesFocusable={!readOnly}
            nodesFocusable={!readOnly}
          >
            {/* Background */}
            {showGrid && (
              <Background
                gap={15}
                size={1}
                color={theme.palette.divider}
              />
            )}
            
            {/* Node Palette */}
            {!readOnly && (
              <Panel position="top-left">
                <NodePalette onDragStart={onNodeDragStart} />
                <Box 
                  sx={{ 
                    mt: 2, 
                    p: 1.5, 
                    bgcolor: 'rgba(255, 255, 255, 0.9)', 
                    borderRadius: 1,
                    boxShadow: 1,
                    maxWidth: 250
                  }}
                >
                  <Typography variant="subtitle2" gutterBottom>
                    How to connect nodes:
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    1. Drag from an output handle (right side)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    2. Drop on an input handle (left side)
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Handles are the colored circles on the sides of each node.
                  </Typography>
                </Box>
              </Panel>
            )}
            
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
              nodeColor={(node) => {
                const nodeConfig = NODE_TYPE_CONFIG[node.type as NodeType];
                return nodeConfig?.color || '#eee';
              }}
            />
          </ReactFlow>
        </Box>
        {/* Node Edit Forms */}
        {renderNodeForm}
      </ReactFlowProvider>
    </Box>
  );
};

export default WorkflowCanvas;
