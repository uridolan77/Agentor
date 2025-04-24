import React, { memo } from 'react';
import { EdgeProps, getStraightPath, EdgeLabelRenderer } from 'reactflow';
import { Box, Chip } from '@mui/material';
import { CallSplit as CallSplitIcon } from '@mui/icons-material';
import BaseEdge, { BaseEdgeProps } from './BaseEdge';
import { EdgeData, EdgeType } from '../types';

/**
 * Control edge component
 * Represents control flow between nodes (e.g., conditional branching)
 * Extends the base edge with control-specific styling and functionality
 */
const ControlEdge: React.FC<BaseEdgeProps> = (props) => {
  const { data, style, ...rest } = props;
  
  // Control-specific styling
  const controlStyle = {
    ...style,
    strokeDasharray: '5,5',
  };
  
  // Control-specific label
  const controlLabel = data?.label || 'Control' as React.ReactNode;
  
  // Control-specific marker end
  const controlMarkerEnd = 'url(#control-arrow)';
  
  // Enhanced data for the base edge
  const enhancedData = {
    ...(data || {}),
    type: EdgeType.CONTROL,
    id: props.id, // Ensure id is always defined
    source: props.source || '',
    target: props.target || '',
    sourceHandle: props.sourceHandleId || '',
    targetHandle: props.targetHandleId || ''
  };
  
  return (
    <>
      {/* Define the arrow marker for control edges */}
      <defs>
        <marker
          id="control-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff9800" />
        </marker>
      </defs>
      
      {/* Render the base edge with control-specific props */}
      <BaseEdge
        {...rest}
        data={enhancedData}
        style={controlStyle}
        label={controlLabel}
        markerEnd={controlMarkerEnd}
        animated={false}
      />
    </>
  );
};

export default memo(ControlEdge);
