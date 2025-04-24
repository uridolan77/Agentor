/**
 * Workflow type definitions for the workflow canvas
 */

import { NodeData } from './nodeTypes';
import { EdgeData } from './edgeTypes';

// Workflow Data
export interface WorkflowData {
  id: string;
  name: string;
  description?: string;
  nodes: NodeData[];
  edges: EdgeData[];
  viewport?: {
    x: number;
    y: number;
    zoom: number;
  };
  version: string;
  metadata?: Record<string, any>;
}

// Workflow Execution Status
export enum ExecutionStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  STOPPED = 'stopped'
}

// Workflow Execution State
export interface WorkflowExecutionState {
  status: ExecutionStatus;
  progress: number;
  startTime?: string;
  endTime?: string;
  currentNodeId?: string;
  error?: string;
  logs: ExecutionLog[];
}

// Execution Log Level
export enum LogLevel {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  SUCCESS = 'success'
}

// Execution Log
export interface ExecutionLog {
  id: string;
  timestamp: string;
  level: LogLevel;
  nodeId?: string;
  nodeName?: string;
  message: string;
  details?: string;
}

// Create a new workflow with default values
export const createWorkflow = (
  name: string,
  description?: string,
  id?: string
): WorkflowData => {
  return {
    id: id || `workflow-${Date.now()}`,
    name,
    description,
    nodes: [],
    edges: [],
    viewport: {
      x: 0,
      y: 0,
      zoom: 1
    },
    version: '1.0.0',
    metadata: {
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
  };
};

// Create a new execution log
export const createLog = (
  level: LogLevel,
  message: string,
  nodeId?: string,
  nodeName?: string,
  details?: string
): ExecutionLog => {
  return {
    id: `log-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    timestamp: new Date().toISOString(),
    level,
    nodeId,
    nodeName,
    message,
    details
  };
};

// Create a new execution state
export const createExecutionState = (): WorkflowExecutionState => {
  return {
    status: ExecutionStatus.IDLE,
    progress: 0,
    logs: []
  };
};
