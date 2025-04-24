import React, { memo } from 'react';
import { EdgeProps, getSmoothStepPath, EdgeLabelRenderer } from 'reactflow';
import { Box, Chip } from '@mui/material';
import { DataObject as DataObjectIcon } from '@mui/icons-material';
import BaseEdge, { BaseEdgeProps } from './BaseEdge';
import { EdgeData, EdgeType } from '../types';

/**
 * Data edge component
 * Represents data flow between nodes
 * Extends the base edge with data-specific styling and functionality
 */
const DataEdge: React.FC<BaseEdgeProps> = (props) => {
  const { data, style, ...rest } = props;
  
  // Data-specific styling
  const dataStyle = {
    ...style,
    animation: props.animated ? 'dashdraw 0.5s linear infinite' : undefined,
  };
  
  // Data-specific label
  const dataLabel = data?.label || 'Data' as React.ReactNode;
  
  // Data-specific marker end
  const dataMarkerEnd = 'url(#data-arrow)';
  
  // Enhanced data for the base edge
  const enhancedData = {
    ...(data || {}),
    type: EdgeType.DATA,
    id: props.id, // Ensure id is always defined
    source: props.source || '',
    target: props.target || '',
    sourceHandle: props.sourceHandleId || '',
    targetHandle: props.targetHandleId || ''
  };
  
  return (
    <>
      {/* Define the arrow marker for data edges */}
      <defs>
        <marker
          id="data-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#2196f3" />
        </marker>
      </defs>
      
      {/* Render the base edge with data-specific props */}
      <BaseEdge
        {...rest}
        data={enhancedData}
        style={dataStyle}
        label={dataLabel}
        markerEnd={dataMarkerEnd}
        animated={true}
      />
    </>
  );
};

export default memo(DataEdge);
