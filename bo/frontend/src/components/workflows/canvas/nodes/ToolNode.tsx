import React, { memo } from 'react';
import { Box, Typography, Chip, Divider, LinearProgress } from '@mui/material';
import { Build as BuildIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';

/**
 * Tool node component
 * Extends the base node with tool-specific functionality and UI
 */
const ToolNode = (props: BaseNodeProps) => {
  const { data } = props;
  const toolConfig = data.config || {};
  
  // Tool-specific content to be rendered inside the base node
  const renderToolContent = () => {
    return (
      <Box className="tool-node-content">
        {toolConfig.toolType && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Tool Type
            </Typography>
            <Chip 
              label={toolConfig.toolType} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {toolConfig.parameters && Object.keys(toolConfig.parameters).length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Parameters
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
              {Object.keys(toolConfig.parameters).map((param, index) => (
                <Chip 
                  key={index}
                  label={`${param}: ${toolConfig.parameters[param]}`}
                  size="small"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          </Box>
        )}
        
        {toolConfig.performance && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Performance
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {toolConfig.performance}%
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={toolConfig.performance} 
                sx={{ mt: 0.5, height: 4, borderRadius: 1 }}
              />
            </Box>
          </Box>
        )}
        
        {toolConfig.lastExecuted && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Last Executed
            </Typography>
            <Typography variant="caption">
              {new Date(toolConfig.lastExecuted).toLocaleString()}
            </Typography>
          </Box>
        )}
      </Box>
    );
  };
  
  // Inject the tool-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderToolContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(ToolNode);
