import React from 'react';
import { Box, useTheme } from '@mui/material';
import { NODE_TYPES } from './WorkflowNode';

export interface EdgeData {
  id: string;
  source: string;
  target: string;
  sourceType: keyof typeof NODE_TYPES;
  targetType: keyof typeof NODE_TYPES;
  label?: string;
  animated?: boolean;
}

interface WorkflowEdgeProps {
  edge: EdgeData;
  sourcePosition: { x: number; y: number };
  targetPosition: { x: number; y: number };
  selected: boolean;
  onSelect: (id: string) => void;
}

/**
 * WorkflowEdge component represents a connection between nodes in the workflow editor.
 * It draws a bezier curve between the source and target nodes.
 */
const WorkflowEdge: React.FC<WorkflowEdgeProps> = ({
  edge,
  sourcePosition,
  targetPosition,
  selected,
  onSelect
}) => {
  const theme = useTheme();
  
  // Calculate edge path
  const sourceX = sourcePosition.x + 200; // Right side of source node
  const sourceY = sourcePosition.y + 40; // Middle of source node
  const targetX = targetPosition.x; // Left side of target node
  const targetY = targetPosition.y + 40; // Middle of target node
  
  // Control points for bezier curve
  const controlPointX1 = sourceX + Math.min(80, (targetX - sourceX) / 2);
  const controlPointX2 = targetX - Math.min(80, (targetX - sourceX) / 2);
  
  // Path for the bezier curve
  const path = `M ${sourceX} ${sourceY} C ${controlPointX1} ${sourceY}, ${controlPointX2} ${targetY}, ${targetX} ${targetY}`;
  
  // Get colors from node types
  const sourceColor = NODE_TYPES[edge.sourceType].color;
  const targetColor = NODE_TYPES[edge.targetType].color;
  
  // Create gradient ID
  const gradientId = `gradient-${edge.id}`;
  
  // Handle edge click
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect(edge.id);
  };

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0
      }}
    >
      <svg
        width="100%"
        height="100%"
        style={{ position: 'absolute', top: 0, left: 0 }}
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={sourceColor} />
            <stop offset="100%" stopColor={targetColor} />
          </linearGradient>
          
          {edge.animated && (
            <marker
              id={`arrowhead-${edge.id}`}
              markerWidth="10"
              markerHeight="7"
              refX="10"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill={targetColor}
              />
            </marker>
          )}
        </defs>
        
        {/* Edge Path */}
        <path
          d={path}
          stroke={`url(#${gradientId})`}
          strokeWidth={selected ? 3 : 2}
          fill="none"
          strokeDasharray={edge.animated ? "5,5" : "none"}
          markerEnd={edge.animated ? `url(#arrowhead-${edge.id})` : "none"}
          style={{
            pointerEvents: 'stroke',
            cursor: 'pointer',
            transition: 'stroke-width 0.2s'
          }}
          onClick={handleClick}
        />
        
        {/* Edge Label */}
        {edge.label && (
          <text
            x={(sourceX + targetX) / 2}
            y={(sourceY + targetY) / 2 - 10}
            textAnchor="middle"
            fill={theme.palette.text.secondary}
            fontSize="12px"
            style={{
              pointerEvents: 'none',
              userSelect: 'none',
              fontFamily: theme.typography.fontFamily
            }}
          >
            {edge.label}
          </text>
        )}
      </svg>
    </Box>
  );
};

export default WorkflowEdge;
