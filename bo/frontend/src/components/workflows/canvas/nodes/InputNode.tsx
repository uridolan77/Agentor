import React, { memo } from 'react';
import { Box, Typography, Chip, Divider, TextField } from '@mui/material';
import { Input as InputIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';

/**
 * Input node component
 * Extends the base node with input-specific functionality and UI
 */
const InputNode = (props: BaseNodeProps) => {
  const { data } = props;
  const inputConfig = data.config || {};
  
  // Input-specific content to be rendered inside the base node
  const renderInputContent = () => {
    return (
      <Box className="input-node-content">
        {inputConfig.dataType && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Data Type
            </Typography>
            <Chip 
              label={inputConfig.dataType} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {inputConfig.required !== undefined && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Required
            </Typography>
            <Chip 
              label={inputConfig.required ? 'Yes' : 'No'} 
              size="small"
              color={inputConfig.required ? 'error' : 'default'}
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {inputConfig.defaultValue !== undefined && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Default Value
            </Typography>
            <Box 
              sx={{ 
                mt: 0.5, 
                p: 0.5, 
                bgcolor: 'background.default', 
                borderRadius: 1,
                fontSize: '0.75rem',
                fontFamily: 'monospace',
                overflowX: 'auto'
              }}
            >
              {typeof inputConfig.defaultValue === 'object' 
                ? JSON.stringify(inputConfig.defaultValue) 
                : String(inputConfig.defaultValue)}
            </Box>
          </Box>
        )}
        
        {inputConfig.validation && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Typography variant="caption" color="text.secondary" display="block">
              Validation
            </Typography>
            {Object.entries(inputConfig.validation).map(([key, value]) => (
              <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  {key}
                </Typography>
                <Typography variant="caption">
                  {String(value)}
                </Typography>
              </Box>
            ))}
          </Box>
        )}
        
        {/* Preview input field */}
        <Box sx={{ mt: 1 }}>
          <Divider sx={{ my: 0.5 }} />
          <TextField
            label="Input Value"
            variant="outlined"
            size="small"
            fullWidth
            disabled
            defaultValue={inputConfig.value || inputConfig.defaultValue || ''}
            sx={{ mt: 0.5, '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
          />
        </Box>
      </Box>
    );
  };
  
  // Inject the input-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderInputContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(InputNode);
