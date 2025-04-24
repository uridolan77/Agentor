/**
 * Edge type definitions for the workflow canvas
 */

// Edge Types
export enum EdgeType {
  DATA = 'data',
  CONTROL = 'control'
}

// Edge Data
export interface EdgeData {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
  type: EdgeType;
  label?: string;
  animated?: boolean;
  style?: Record<string, any>;
  data?: Record<string, any>;
}

// Edge Type Configuration
export const EDGE_TYPE_CONFIG = {
  [EdgeType.DATA]: {
    label: 'Data Flow',
    color: '#2196f3',
    description: 'Represents data flowing between nodes',
    animated: true,
    dashed: false
  },
  [EdgeType.CONTROL]: {
    label: 'Control Flow',
    color: '#ff9800',
    description: 'Represents control flow between nodes',
    animated: false,
    dashed: true
  }
};

// Create a new edge with default values
export const createEdge = (
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string,
  type: EdgeType = EdgeType.DATA,
  id?: string
): EdgeData => {
  const config = EDGE_TYPE_CONFIG[type];
  
  return {
    id: id || `edge-${Date.now()}`,
    source,
    sourceHandle,
    target,
    targetHandle,
    type,
    animated: config.animated,
    style: {
      stroke: config.color,
      strokeDasharray: config.dashed ? '5,5' : undefined
    },
    data: {
      edgeType: type
    }
  };
};

// Validate if a connection is valid based on port types
export const isValidConnection = (
  sourceType: string,
  targetType: string
): boolean => {
  // If either type is 'any', the connection is valid
  if (sourceType === 'any' || targetType === 'any') {
    return true;
  }
  
  // Otherwise, the types must match
  return sourceType === targetType;
};
