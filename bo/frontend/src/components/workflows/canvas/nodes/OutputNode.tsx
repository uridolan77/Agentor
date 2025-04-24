import React, { memo } from 'react';
import { Box, Typography, Chip, Divider, Paper } from '@mui/material';
import { Output as OutputIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';

/**
 * Output node component
 * Extends the base node with output-specific functionality and UI
 */
const OutputNode = (props: BaseNodeProps) => {
  const { data } = props;
  const outputConfig = data.config || {};
  
  // Format output value for display
  const formatOutputValue = (value: any) => {
    if (value === undefined || value === null) {
      return 'No data';
    }
    
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value, null, 2);
      } catch (e) {
        return String(value);
      }
    }
    
    return String(value);
  };
  
  // Output-specific content to be rendered inside the base node
  const renderOutputContent = () => {
    return (
      <Box className="output-node-content">
        {outputConfig.dataType && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Data Type
            </Typography>
            <Chip 
              label={outputConfig.dataType} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {outputConfig.destination && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Destination
            </Typography>
            <Chip 
              label={outputConfig.destination} 
              size="small"
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {outputConfig.format && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Format
            </Typography>
            <Chip 
              label={outputConfig.format} 
              size="small"
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {/* Output value preview */}
        <Box sx={{ mt: 1 }}>
          <Divider sx={{ my: 0.5 }} />
          <Typography variant="caption" color="text.secondary" display="block">
            Output Value
          </Typography>
          <Paper
            variant="outlined"
            sx={{ 
              mt: 0.5, 
              p: 1, 
              maxHeight: 100, 
              overflow: 'auto',
              bgcolor: 'background.default',
              fontSize: '0.75rem',
              fontFamily: 'monospace',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word'
            }}
          >
            {formatOutputValue(outputConfig.value)}
          </Paper>
        </Box>
        
        {outputConfig.status && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Status
              </Typography>
              <Chip 
                label={outputConfig.status} 
                size="small"
                color={
                  outputConfig.status === 'success' ? 'success' :
                  outputConfig.status === 'error' ? 'error' :
                  outputConfig.status === 'pending' ? 'warning' :
                  'default'
                }
                sx={{ fontSize: '0.7rem' }}
              />
            </Box>
          </Box>
        )}
      </Box>
    );
  };
  
  // Inject the output-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderOutputContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(OutputNode);
