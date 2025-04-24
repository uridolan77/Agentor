import React from 'react';
import { NodeTypes } from 'reactflow';
import { NodeType } from '../types';
import BaseNode from './BaseNode';
import AgentNode from './AgentNode';
import ToolNode from './ToolNode';
import ConditionNode from './ConditionNode';
import InputNode from './InputNode';
import OutputNode from './OutputNode';
import GroupNode from './GroupNode';
import ConsensusNode from './ConsensusNode';
import TemporalCoordinationNode from './TemporalCoordinationNode';
import DatabaseNode from './DatabaseNode';
import FilesystemNode from './FilesystemNode';
import APINode from './APINode';

/**
 * Node factory for creating node components based on node type
 * Maps node types to their respective components
 */
export const nodeTypes: NodeTypes = {
  [NodeType.AGENT]: AgentNode,
  [NodeType.TOOL]: ToolNode,
  [NodeType.CONDITION]: ConditionNode,
  [NodeType.INPUT]: InputNode,
  [NodeType.OUTPUT]: OutputNode,
  [NodeType.GROUP]: GroupNode,
  [NodeType.CONSENSUS]: ConsensusNode,
  [NodeType.TEMPORAL_COORDINATION]: TemporalCoordinationNode,
  [NodeType.DATABASE]: DatabaseNode,
  [NodeType.FILESYSTEM]: FilesystemNode,
  [NodeType.API]: APINode,
  default: BaseNode
};

/**
 * Get the appropriate node component for a given node type
 * @param type Node type
 * @returns Node component
 */
export const getNodeComponent = (type: NodeType) => {
  return nodeTypes[type] || nodeTypes.default;
};

export default nodeTypes;
