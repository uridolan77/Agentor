import React from 'react';
import { EdgeTypes } from 'reactflow';
import { EdgeType } from '../types';
import BaseEdge from './BaseEdge';
import DataEdge from './DataEdge';
import ControlEdge from './ControlEdge';

/**
 * Edge factory for creating edge components based on edge type
 * Maps edge types to their respective components
 */
export const edgeTypes: EdgeTypes = {
  [EdgeType.DATA]: DataEdge,
  [EdgeType.CONTROL]: ControlEdge,
  default: BaseEdge
};

/**
 * Get the appropriate edge component for a given edge type
 * @param type Edge type
 * @returns Edge component
 */
export const getEdgeComponent = (type: EdgeType) => {
  return edgeTypes[type] || edgeTypes.default;
};

export default edgeTypes;
