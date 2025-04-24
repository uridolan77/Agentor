import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, IconButton, Tooltip, useTheme } from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Add as AddIcon
} from '@mui/icons-material';
import WorkflowNode, { NodeData, NODE_TYPES } from './WorkflowNode';
import WorkflowEdge, { EdgeData } from './WorkflowEdge';
import NodeToolbar from './NodeToolbar';

export interface WorkflowData {
  nodes: NodeData[];
  edges: EdgeData[];
}

interface WorkflowCanvasProps {
  workflow: WorkflowData;
  onWorkflowChange: (workflow: WorkflowData) => void;
  readOnly?: boolean;
}

/**
 * WorkflowCanvas component is the main component for the workflow editor.
 * It renders nodes and edges, and handles interactions like dragging, zooming, and selecting.
 */
const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  workflow,
  onWorkflowChange,
  readOnly = false
}) => {
  const theme = useTheme();
  const canvasRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [showNodeToolbar, setShowNodeToolbar] = useState(false);
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });

  // Get selected node
  const selectedNode = workflow.nodes.find(node => node.id === selectedNodeId);

  // Handle canvas mouse down
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0 || readOnly) return; // Only left mouse button
    
    // Clear selection
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setShowNodeToolbar(false);
    
    // Start canvas dragging
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    
    // Add global event listeners
    document.addEventListener('mousemove', handleCanvasMouseMove);
    document.addEventListener('mouseup', handleCanvasMouseUp);
  };

  // Handle canvas mouse move
  const handleCanvasMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    // Calculate new position
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    
    setPosition(prev => ({
      x: prev.x + dx,
      y: prev.y + dy
    }));
    
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  // Handle canvas mouse up
  const handleCanvasMouseUp = () => {
    setIsDragging(false);
    
    // Remove global event listeners
    document.removeEventListener('mousemove', handleCanvasMouseMove);
    document.removeEventListener('mouseup', handleCanvasMouseUp);
  };

  // Handle zoom in
  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.1, 2));
  };

  // Handle zoom out
  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.1, 0.5));
  };

  // Handle reset view
  const handleResetView = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  // Handle node select
  const handleNodeSelect = (id: string) => {
    setSelectedNodeId(id);
    setSelectedEdgeId(null);
    
    // Show node toolbar
    if (!readOnly) {
      const node = workflow.nodes.find(n => n.id === id);
      if (node) {
        setToolbarPosition({
          x: node.position.x + 100, // Center of node
          y: node.position.y - 40 // Above node
        });
        setShowNodeToolbar(true);
      }
    }
  };

  // Handle edge select
  const handleEdgeSelect = (id: string) => {
    setSelectedEdgeId(id);
    setSelectedNodeId(null);
    setShowNodeToolbar(false);
  };

  // Handle node move
  const handleNodeMove = (id: string, newPosition: { x: number; y: number }) => {
    const updatedNodes = workflow.nodes.map(node => 
      node.id === id ? { ...node, position: newPosition } : node
    );
    
    onWorkflowChange({
      ...workflow,
      nodes: updatedNodes
    });
  };

  // Handle node edit
  const handleNodeEdit = (id: string) => {
    // This would open a modal or panel to edit the node
    console.log('Edit node:', id);
  };

  // Handle node delete
  const handleNodeDelete = (id: string) => {
    // Remove node and connected edges
    const updatedNodes = workflow.nodes.filter(node => node.id !== id);
    const updatedEdges = workflow.edges.filter(
      edge => edge.source !== id && edge.target !== id
    );
    
    onWorkflowChange({
      nodes: updatedNodes,
      edges: updatedEdges
    });
    
    setSelectedNodeId(null);
    setShowNodeToolbar(false);
  };

  // Handle node duplicate
  const handleNodeDuplicate = (id: string) => {
    const nodeToDuplicate = workflow.nodes.find(node => node.id === id);
    if (!nodeToDuplicate) return;
    
    // Create new node with unique ID
    const newNode: NodeData = {
      ...nodeToDuplicate,
      id: `node-${Date.now()}`,
      position: {
        x: nodeToDuplicate.position.x + 20,
        y: nodeToDuplicate.position.y + 20
      }
    };
    
    onWorkflowChange({
      ...workflow,
      nodes: [...workflow.nodes, newNode]
    });
  };

  // Handle node run
  const handleNodeRun = (id: string) => {
    // This would trigger execution of just this node
    console.log('Run node:', id);
  };

  // Handle add node
  const handleAddNode = (type: keyof typeof NODE_TYPES, x: number, y: number) => {
    // Create new node
    const newNode: NodeData = {
      id: `node-${Date.now()}`,
      type: type,
      label: `New ${String(type)}`,
      description: '',
      position: { x, y }
    };
    
    onWorkflowChange({
      ...workflow,
      nodes: [...workflow.nodes, newNode]
    });
    
    setShowNodeToolbar(false);
  };

  // Handle add edge
  const handleAddEdge = (source: string, target: string) => {
    // Find source and target nodes
    const sourceNode = workflow.nodes.find(node => node.id === source);
    const targetNode = workflow.nodes.find(node => node.id === target);
    
    if (!sourceNode || !targetNode) return;
    
    // Create new edge
    const newEdge: EdgeData = {
      id: `edge-${Date.now()}`,
      source,
      target,
      sourceType: sourceNode.type,
      targetType: targetNode.type
    };
    
    onWorkflowChange({
      ...workflow,
      edges: [...workflow.edges, newEdge]
    });
  };

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (readOnly) return;
      
      // Delete selected node or edge
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedNodeId) {
          handleNodeDelete(selectedNodeId);
        } else if (selectedEdgeId) {
          const updatedEdges = workflow.edges.filter(edge => edge.id !== selectedEdgeId);
          onWorkflowChange({
            ...workflow,
            edges: updatedEdges
          });
          setSelectedEdgeId(null);
        }
      }
      
      // Escape to clear selection
      if (e.key === 'Escape') {
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
        setShowNodeToolbar(false);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [workflow, selectedNodeId, selectedEdgeId, readOnly, onWorkflowChange]);

  return (
    <Paper
      sx={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        bgcolor: theme.palette.background.default
      }}
    >
      {/* Canvas */}
      <Box
        ref={canvasRef}
        id="workflow-canvas"
        sx={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          backgroundImage: 'radial-gradient(circle, #e0e0e0 1px, transparent 1px)',
          backgroundSize: '20px 20px',
          transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`,
          transformOrigin: '0 0',
          cursor: isDragging ? 'grabbing' : 'grab'
        }}
        onMouseDown={handleCanvasMouseDown}
      >
        {/* Edges */}
        {workflow.edges.map(edge => {
          const sourceNode = workflow.nodes.find(node => node.id === edge.source);
          const targetNode = workflow.nodes.find(node => node.id === edge.target);
          
          if (!sourceNode || !targetNode) return null;
          
          return (
            <WorkflowEdge
              key={edge.id}
              edge={edge}
              sourcePosition={sourceNode.position}
              targetPosition={targetNode.position}
              selected={edge.id === selectedEdgeId}
              onSelect={handleEdgeSelect}
            />
          );
        })}
        
        {/* Nodes */}
        {workflow.nodes.map(node => (
          <WorkflowNode
            key={node.id}
            node={node}
            selected={node.id === selectedNodeId}
            onSelect={handleNodeSelect}
            onMove={handleNodeMove}
            onEdit={handleNodeEdit}
            onDelete={handleNodeDelete}
            onDuplicate={handleNodeDuplicate}
            onRun={handleNodeRun}
          />
        ))}
        
        {/* Node Toolbar */}
        {showNodeToolbar && selectedNode && (
          <NodeToolbar
            position={toolbarPosition}
            onAddNode={(type: keyof typeof NODE_TYPES) => {
              handleAddNode(
                type,
                selectedNode.position.x + 250,
                selectedNode.position.y
              );
            }}
          />
        )}
      </Box>
      
      {/* Controls */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          zIndex: 10
        }}
      >
        <Tooltip title="Zoom In">
          <IconButton
            onClick={handleZoomIn}
            sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
          >
            <ZoomInIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Zoom Out">
          <IconButton
            onClick={handleZoomOut}
            sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
          >
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Reset View">
          <IconButton
            onClick={handleResetView}
            sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
          >
            <CenterIcon />
          </IconButton>
        </Tooltip>
        
        {!readOnly && (
          <Tooltip title="Add Node">
            <IconButton
              onClick={() => {
                // Calculate center of visible canvas
                if (!canvasRef.current) return;
                
                const rect = canvasRef.current.getBoundingClientRect();
                const centerX = (rect.width / 2 - position.x) / scale;
                const centerY = (rect.height / 2 - position.y) / scale;
                
                handleAddNode('AGENT', centerX - 100, centerY - 40);
              }}
              sx={{ bgcolor: 'primary.main', color: 'white', boxShadow: 1 }}
            >
              <AddIcon />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </Paper>
  );
};

export default WorkflowCanvas;
