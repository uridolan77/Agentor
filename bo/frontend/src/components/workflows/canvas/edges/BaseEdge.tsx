import React, { memo } from 'react';
import { EdgeProps, getBezierPath, EdgeLabelRenderer } from 'reactflow';
import { Box, Typography, IconButton, Tooltip } from '@mui/material';
import { Delete as DeleteIcon, Settings as SettingsIcon } from '@mui/icons-material';
import { EdgeData, EdgeType, EDGE_TYPE_CONFIG } from '../types';

export interface BaseEdgeProps extends EdgeProps<EdgeData> {
  selected?: boolean;
  animated?: boolean;
  label?: React.ReactNode;
  labelStyle?: React.CSSProperties;
  onEdgeClick?: (id: string, event: React.MouseEvent) => void;
  onEdgeDelete?: (id: string) => void;
  onEdgeEdit?: (id: string) => void;
}

/**
 * Base edge component for all edge types
 * Renders a bezier curve between two nodes with optional label and controls
 */
const BaseEdge: React.FC<BaseEdgeProps> = ({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  animated = false,
  data,
  style = {},
  label,
  labelStyle = {},
  onEdgeClick,
  onEdgeDelete,
  onEdgeEdit,
  markerEnd
}) => {
  // Get edge configuration based on type
  const edgeType = data?.type || EdgeType.DATA;
  const edgeConfig = EDGE_TYPE_CONFIG[edgeType];
  
  // Calculate the path
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  
  // Edge style based on type and selection
  const edgeStyle = {
    stroke: edgeConfig.color,
    strokeWidth: selected ? 3 : 2,
    strokeDasharray: edgeConfig.dashed ? '5,5' : undefined,
    ...style,
  };
  
  // Handle edge click
  const handleEdgeClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (onEdgeClick) {
      onEdgeClick(id, event);
    }
  };
  
  // Handle edge delete
  const handleDelete = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (onEdgeDelete) {
      onEdgeDelete(id);
    }
  };
  
  // Handle edge edit
  const handleEdit = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (onEdgeEdit) {
      onEdgeEdit(id);
    }
  };
  
  // Determine if the edge should be animated
  const isAnimated = animated || edgeConfig.animated;
  
  // Get the label text
  const labelText = label || data?.label || edgeConfig.label;
  
  return (
    <>
      <path
        id={id}
        className={`react-flow__edge-path ${isAnimated ? 'animated' : ''}`}
        d={edgePath}
        style={edgeStyle}
        markerEnd={markerEnd}
        onClick={handleEdgeClick}
      />
      
      {/* Edge Label */}
      {labelText && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
              backgroundColor: 'white',
              padding: '3px 5px',
              borderRadius: '4px',
              fontSize: 12,
              fontWeight: 500,
              border: '1px solid #ccc',
              ...labelStyle
            }}
            className="nodrag nopan"
          >
            {labelText}
          </div>
        </EdgeLabelRenderer>
      )}
      
      {/* Edge Controls (visible when selected) */}
      {selected && (onEdgeEdit || onEdgeDelete) && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY - 25}px)`,
              pointerEvents: 'all',
              display: 'flex',
              gap: '4px',
              zIndex: 10
            }}
            className="nodrag nopan"
          >
            {onEdgeEdit && (
              <Tooltip title="Edit Edge">
                <IconButton
                  size="small"
                  sx={{ 
                    bgcolor: 'background.paper', 
                    boxShadow: 1,
                    width: 24,
                    height: 24
                  }}
                  onClick={handleEdit}
                >
                  <SettingsIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            {onEdgeDelete && (
              <Tooltip title="Delete Edge">
                <IconButton
                  size="small"
                  sx={{ 
                    bgcolor: 'background.paper', 
                    boxShadow: 1,
                    width: 24,
                    height: 24,
                    color: 'error.main'
                  }}
                  onClick={handleDelete}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(BaseEdge);
