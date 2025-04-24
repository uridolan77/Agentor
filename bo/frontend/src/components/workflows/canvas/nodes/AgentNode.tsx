import React, { memo } from 'react';
import { NodeProps } from 'reactflow';
import { Box, Typography, Chip, Divider } from '@mui/material';
import { SmartToy as SmartToyIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';

/**
 * Agent node component
 * Extends the base node with agent-specific functionality and UI
 */
const AgentNode = (props: BaseNodeProps) => {
  const { data } = props;
  const agentConfig = data.config || {};
  
  // Agent-specific content to be rendered inside the base node
  const renderAgentContent = () => {
    return (
      <Box className="agent-node-content">
        {agentConfig.model && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Model
            </Typography>
            <Chip 
              label={agentConfig.model} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {agentConfig.capabilities && agentConfig.capabilities.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Capabilities
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
              {agentConfig.capabilities.map((capability: string, index: number) => (
                <Chip 
                  key={index}
                  label={capability}
                  size="small"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          </Box>
        )}
        
        {agentConfig.status && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="caption" color="text.secondary">
                Status
              </Typography>
              <Chip 
                label={agentConfig.status} 
                size="small"
                color={agentConfig.status === 'active' ? 'success' : 'default'}
                sx={{ fontSize: '0.7rem' }}
              />
            </Box>
          </Box>
        )}
      </Box>
    );
  };
  
  // Inject the agent-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderAgentContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(AgentNode);
