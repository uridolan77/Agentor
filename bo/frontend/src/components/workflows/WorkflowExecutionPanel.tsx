import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Button,
  IconButton,
  Tooltip,
  Collapse,
  CircularProgress,
  Chip,
  useTheme
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Save as SaveIcon,
  Download as DownloadIcon,
  Clear as ClearIcon
} from '@mui/icons-material';

export interface ExecutionLog {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  nodeId?: string;
  nodeName?: string;
  message: string;
  details?: string;
}

export interface ExecutionStatus {
  status: 'idle' | 'running' | 'completed' | 'failed' | 'stopped';
  progress: number;
  startTime?: string;
  endTime?: string;
  currentNodeId?: string;
  error?: string;
}

interface WorkflowExecutionPanelProps {
  workflowId: string;
  workflowName: string;
  status: ExecutionStatus;
  logs: ExecutionLog[];
  onStart: () => void;
  onStop: () => void;
  onClearLogs: () => void;
  onSaveExecution: () => void;
  onDownloadLogs: () => void;
}

/**
 * WorkflowExecutionPanel component displays the execution status and logs of a workflow.
 * It allows starting, stopping, and monitoring workflow execution.
 */
const WorkflowExecutionPanel: React.FC<WorkflowExecutionPanelProps> = ({
  workflowId,
  workflowName,
  status,
  logs,
  onStart,
  onStop,
  onClearLogs,
  onSaveExecution,
  onDownloadLogs
}) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(true);

  // Toggle panel expansion
  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  // Get status color
  const getStatusColor = () => {
    switch (status.status) {
      case 'running':
        return theme.palette.info.main;
      case 'completed':
        return theme.palette.success.main;
      case 'failed':
        return theme.palette.error.main;
      case 'stopped':
        return theme.palette.warning.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  // Get log level color
  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'info':
        return theme.palette.info.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'success':
        return theme.palette.success.main;
      default:
        return theme.palette.text.primary;
    }
  };

  // Calculate execution time
  const getExecutionTime = () => {
    if (!status.startTime) return '0s';
    
    const start = new Date(status.startTime).getTime();
    const end = status.endTime 
      ? new Date(status.endTime).getTime() 
      : new Date().getTime();
    
    const seconds = Math.floor((end - start) / 1000);
    
    if (seconds < 60) {
      return `${seconds}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  return (
    <Paper
      sx={{
        position: 'relative',
        width: '100%',
        borderRadius: 1,
        overflow: 'hidden',
        boxShadow: 2
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          bgcolor: theme.palette.background.paper,
          p: 1,
          borderBottom: expanded ? `1px solid ${theme.palette.divider}` : 'none'
        }}
      >
        <Box display="flex" alignItems="center">
          <Typography variant="subtitle1" fontWeight="bold" sx={{ mr: 2 }}>
            Execution Panel
          </Typography>
          <Chip
            label={status.status.toUpperCase()}
            size="small"
            sx={{
              bgcolor: `${getStatusColor()}20`,
              color: getStatusColor(),
              fontWeight: 'bold'
            }}
          />
        </Box>
        <Box display="flex" alignItems="center">
          {status.status === 'running' && (
            <CircularProgress
              size={20}
              variant="determinate"
              value={status.progress}
              sx={{ mr: 1 }}
            />
          )}
          <IconButton size="small" onClick={toggleExpanded}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>
      </Box>

      {/* Collapsible Content */}
      <Collapse in={expanded}>
        {/* Controls */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            bgcolor: theme.palette.background.default,
            p: 1
          }}
        >
          <Box>
            <Button
              variant="contained"
              color="primary"
              startIcon={<PlayIcon />}
              onClick={onStart}
              disabled={status.status === 'running'}
              size="small"
              sx={{ mr: 1 }}
            >
              Run
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<StopIcon />}
              onClick={onStop}
              disabled={status.status !== 'running'}
              size="small"
            >
              Stop
            </Button>
          </Box>
          <Box>
            <Tooltip title="Clear Logs">
              <IconButton size="small" onClick={onClearLogs} sx={{ mr: 0.5 }}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Save Execution">
              <IconButton size="small" onClick={onSaveExecution} sx={{ mr: 0.5 }}>
                <SaveIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download Logs">
              <IconButton size="small" onClick={onDownloadLogs}>
                <DownloadIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Status Info */}
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 2,
            p: 1,
            bgcolor: theme.palette.background.paper,
            borderTop: `1px solid ${theme.palette.divider}`,
            borderBottom: `1px solid ${theme.palette.divider}`
          }}
        >
          <Box>
            <Typography variant="caption" color="text.secondary">
              Workflow
            </Typography>
            <Typography variant="body2">{workflowName}</Typography>
          </Box>
          {status.startTime && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Started
              </Typography>
              <Typography variant="body2">
                {formatTimestamp(status.startTime)}
              </Typography>
            </Box>
          )}
          {status.endTime && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Ended
              </Typography>
              <Typography variant="body2">
                {formatTimestamp(status.endTime)}
              </Typography>
            </Box>
          )}
          {status.startTime && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Duration
              </Typography>
              <Typography variant="body2">{getExecutionTime()}</Typography>
            </Box>
          )}
          {status.currentNodeId && (
            <Box>
              <Typography variant="caption" color="text.secondary">
                Current Node
              </Typography>
              <Typography variant="body2">{status.currentNodeId}</Typography>
            </Box>
          )}
        </Box>

        {/* Logs */}
        <Box
          sx={{
            height: 200,
            overflow: 'auto',
            bgcolor: theme.palette.background.default,
            p: 1
          }}
        >
          {logs.length === 0 ? (
            <Typography
              variant="body2"
              color="text.secondary"
              align="center"
              sx={{ py: 4 }}
            >
              No logs available. Run the workflow to see execution logs.
            </Typography>
          ) : (
            logs.map(log => (
              <Box
                key={log.id}
                sx={{
                  mb: 1,
                  p: 1,
                  borderRadius: 1,
                  bgcolor: theme.palette.background.paper
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mb: 0.5
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{ color: getLogLevelColor(log.level) }}
                  >
                    {log.level.toUpperCase()}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatTimestamp(log.timestamp)}
                  </Typography>
                </Box>
                {log.nodeId && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    display="block"
                    mb={0.5}
                  >
                    Node: {log.nodeName || log.nodeId}
                  </Typography>
                )}
                <Typography variant="body2">{log.message}</Typography>
                {log.details && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{
                      display: 'block',
                      mt: 0.5,
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace'
                    }}
                  >
                    {log.details}
                  </Typography>
                )}
              </Box>
            ))
          )}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default WorkflowExecutionPanel;
