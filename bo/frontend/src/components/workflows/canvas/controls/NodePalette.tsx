import React from 'react';
import { Paper, Box, Typography, Tabs, Tab, Tooltip, useTheme } from '@mui/material';
import {
  SmartToy as AgentIcon,
  Build as ToolIcon,
  CallSplit as ConditionIcon,
  Input as InputIcon,
  Output as OutputIcon,
  Folder as GroupIcon,
  HowToVote as ConsensusIcon,
  AccessTime as TemporalCoordinationIcon,
  Storage as DatabaseIcon,
  Folder as FilesystemIcon,
  Http as APIIcon
} from '@mui/icons-material';
import { NodeType, NODE_TYPE_CONFIG } from '../types';

interface NodePaletteProps {
  onDragStart: (event: React.DragEvent<HTMLDivElement>, nodeType: NodeType) => void;
}

/**
 * NodePalette component
 * Displays a palette of node types that can be dragged onto the canvas
 */
const NodePalette: React.FC<NodePaletteProps> = ({ onDragStart }) => {
  const theme = useTheme();
  const [tabValue, setTabValue] = React.useState(0);

  // Node types to display in the palette
  const nodeTypes = [
    {
      type: NodeType.AGENT,
      icon: <AgentIcon />,
    },
    {
      type: NodeType.TOOL,
      icon: <ToolIcon />,
    },
    {
      type: NodeType.CONDITION,
      icon: <ConditionIcon />,
    },
    {
      type: NodeType.INPUT,
      icon: <InputIcon />,
    },
    {
      type: NodeType.OUTPUT,
      icon: <OutputIcon />,
    },
    {
      type: NodeType.GROUP,
      icon: <GroupIcon />,
    },
    {
      type: NodeType.CONSENSUS,
      icon: <ConsensusIcon />,
    },
    {
      type: NodeType.TEMPORAL_COORDINATION,
      icon: <TemporalCoordinationIcon />,
    },
    {
      type: NodeType.DATABASE,
      icon: <DatabaseIcon />,
    },
    {
      type: NodeType.FILESYSTEM,
      icon: <FilesystemIcon />,
    },
    {
      type: NodeType.API,
      icon: <APIIcon />,
    }
  ];

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle drag start
  const handleDragStart = (event: React.DragEvent<HTMLDivElement>, nodeType: NodeType) => {
    onDragStart(event, nodeType);
  };

  return (
    <Paper 
      elevation={2}
      sx={{ 
        width: 250,
        borderRadius: 1,
        overflow: 'hidden',
        mb: 2
      }}
    >
      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        variant="fullWidth"
        indicatorColor="primary"
        textColor="primary"
      >
        <Tab label="Nodes" />
        <Tab label="Templates" />
      </Tabs>

      {tabValue === 0 && (
        <Box sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Drag nodes to the canvas
          </Typography>
          <Box 
            sx={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: 1.5,
              mt: 1
            }}
          >
            {nodeTypes.map((nodeType) => {
              const config = NODE_TYPE_CONFIG[nodeType.type];
              return (
                <Tooltip 
                  key={nodeType.type} 
                  title={config.description}
                  placement="top"
                >
                  <Paper
                    elevation={1}
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      p: 1,
                      cursor: 'grab',
                      transition: 'all 0.2s',
                      '&:hover': {
                        boxShadow: theme.shadows[3],
                        transform: 'translateY(-2px)'
                      }
                    }}
                    draggable
                    onDragStart={(e) => handleDragStart(e, nodeType.type)}
                  >
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        bgcolor: `${config.color}20`,
                        color: config.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 0.5
                      }}
                    >
                      {nodeType.icon}
                    </Box>
                    <Typography variant="caption" align="center">
                      {config.label}
                    </Typography>
                  </Paper>
                </Tooltip>
              );
            })}
          </Box>
        </Box>
      )}

      {tabValue === 1 && (
        <Box sx={{ p: 2, height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            Workflow templates will be available soon
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default NodePalette;
