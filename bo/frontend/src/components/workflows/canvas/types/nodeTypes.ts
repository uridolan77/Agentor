/**
 * Node type definitions for the workflow canvas
 */

// Node Types
export enum NodeType {
  AGENT = 'agent',
  TOOL = 'tool',
  CONDITION = 'condition',
  INPUT = 'input',
  OUTPUT = 'output',
  GROUP = 'group',
  // Coordination types
  CONSENSUS = 'consensus',
  TEMPORAL_COORDINATION = 'temporal_coordination',
  // Interface types
  DATABASE = 'database',
  FILESYSTEM = 'filesystem',
  API = 'api'
}

// Port Definition
export interface PortDefinition {
  id: string;
  type: string;
  label?: string;
  validConnections?: string[]; // Valid port types to connect to
}

// Node Data
export interface NodeData {
  id: string;
  type: NodeType;
  label: string;
  description?: string;
  config: Record<string, any>;
  position: { x: number; y: number };
  size?: { width: number; height: number };
  style?: Record<string, any>;
  parentId?: string; // For grouped nodes
  ports?: {
    inputs: PortDefinition[];
    outputs: PortDefinition[];
  };
  renderContent?: () => React.ReactNode; // Function to render custom content
  onNodeEdit?: (nodeId: string) => void; // Function to edit the node
  onNodeDelete?: (nodeId: string) => void; // Function to delete the node
}

// Node Type Configuration
export const NODE_TYPE_CONFIG = {
  [NodeType.AGENT]: {
    label: 'Agent',
    color: '#4caf50',
    description: 'Autonomous agent that performs tasks'
  },
  [NodeType.TOOL]: {
    label: 'Tool',
    color: '#2196f3',
    description: 'Tool that performs specific operations'
  },
  [NodeType.CONDITION]: {
    label: 'Condition',
    color: '#ff9800',
    description: 'Conditional logic for workflow branching'
  },
  [NodeType.INPUT]: {
    label: 'Input',
    color: '#9c27b0',
    description: 'Input data for the workflow'
  },
  [NodeType.OUTPUT]: {
    label: 'Output',
    color: '#f44336',
    description: 'Output data from the workflow'
  },
  [NodeType.GROUP]: {
    label: 'Group',
    color: '#607d8b',
    description: 'Group of nodes'
  },
  // Coordination types
  [NodeType.CONSENSUS]: {
    label: 'Consensus',
    color: '#e91e63',
    description: 'Collective decision making among agents'
  },
  [NodeType.TEMPORAL_COORDINATION]: {
    label: 'Temporal Coordination',
    color: '#673ab7',
    description: 'Time-aware agent coordination'
  },
  // Interface types
  [NodeType.DATABASE]: {
    label: 'Database',
    color: '#3f51b5',
    description: 'Database operations'
  },
  [NodeType.FILESYSTEM]: {
    label: 'Filesystem',
    color: '#009688',
    description: 'File system operations'
  },
  [NodeType.API]: {
    label: 'API',
    color: '#ff5722',
    description: 'API operations'
  }
};

// Default node size
export const DEFAULT_NODE_SIZE = {
  width: 200,
  height: 100
};

// Default node ports
export const DEFAULT_NODE_PORTS = {
  [NodeType.AGENT]: {
    inputs: [
      { id: 'input-default', type: 'any', label: 'Input' }
    ],
    outputs: [
      { id: 'output-default', type: 'any', label: 'Output' }
    ]
  },
  [NodeType.TOOL]: {
    inputs: [
      { id: 'input-default', type: 'any', label: 'Input' }
    ],
    outputs: [
      { id: 'output-default', type: 'any', label: 'Output' }
    ]
  },
  [NodeType.CONDITION]: {
    inputs: [
      { id: 'input-default', type: 'any', label: 'Input' }
    ],
    outputs: [
      { id: 'output-true', type: 'boolean', label: 'True' },
      { id: 'output-false', type: 'boolean', label: 'False' }
    ]
  },
  [NodeType.INPUT]: {
    inputs: [
      { id: 'input-config', type: 'config', label: 'Config' }
    ],
    outputs: [
      { id: 'output-default', type: 'any', label: 'Output' }
    ]
  },
  [NodeType.OUTPUT]: {
    inputs: [
      { id: 'input-default', type: 'any', label: 'Input' }
    ],
    outputs: [
      { id: 'output-result', type: 'result', label: 'Result' }
    ]
  },
  [NodeType.GROUP]: {
    inputs: [
      { id: 'input-group', type: 'any', label: 'Input' }
    ],
    outputs: [
      { id: 'output-group', type: 'any', label: 'Output' }
    ]
  },
  // Coordination types
  [NodeType.CONSENSUS]: {
    inputs: [
      { id: 'input-agents', type: 'agent', label: 'Agents' },
      { id: 'input-options', type: 'any', label: 'Options' }
    ],
    outputs: [
      { id: 'output-decision', type: 'any', label: 'Decision' }
    ]
  },
  [NodeType.TEMPORAL_COORDINATION]: {
    inputs: [
      { id: 'input-agents', type: 'agent', label: 'Agents' },
      { id: 'input-task', type: 'string', label: 'Task' }
    ],
    outputs: [
      { id: 'output-result', type: 'any', label: 'Result' }
    ]
  },
  // Interface types
  [NodeType.DATABASE]: {
    inputs: [
      { id: 'input-query', type: 'string', label: 'Query' }
    ],
    outputs: [
      { id: 'output-result', type: 'any', label: 'Result' }
    ]
  },
  [NodeType.FILESYSTEM]: {
    inputs: [
      { id: 'input-path', type: 'string', label: 'Path' },
      { id: 'input-operation', type: 'string', label: 'Operation' }
    ],
    outputs: [
      { id: 'output-result', type: 'any', label: 'Result' }
    ]
  },
  [NodeType.API]: {
    inputs: [
      { id: 'input-request', type: 'object', label: 'Request' }
    ],
    outputs: [
      { id: 'output-response', type: 'object', label: 'Response' }
    ]
  }
};

// Create a new node with default values
export const createNode = (
  type: NodeType,
  position: { x: number; y: number },
  id?: string
): NodeData => {
  // Make sure we have valid ports for this node type
  const ports = DEFAULT_NODE_PORTS[type] ? { ...DEFAULT_NODE_PORTS[type] } : {
    inputs: [{ id: 'input-default', type: 'any', label: 'Input' }],
    outputs: [{ id: 'output-default', type: 'any', label: 'Output' }]
  };
  
  return {
    id: id || `node-${Date.now()}`,
    type,
    label: `New ${NODE_TYPE_CONFIG[type].label}`,
    description: '',
    config: {},
    position,
    size: DEFAULT_NODE_SIZE,
    ports
  };
};
