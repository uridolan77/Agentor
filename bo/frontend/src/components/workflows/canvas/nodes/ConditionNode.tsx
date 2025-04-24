import React, { memo } from 'react';
import { Box, Typography, Chip, Divider, Switch, FormControlLabel } from '@mui/material';
import { CallSplit as CallSplitIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';

/**
 * Condition node component
 * Extends the base node with condition-specific functionality and UI
 */
const ConditionNode = (props: BaseNodeProps) => {
  const { data } = props;
  const conditionConfig = data.config || {};
  
  // Condition-specific content to be rendered inside the base node
  const renderConditionContent = () => {
    return (
      <Box className="condition-node-content">
        {conditionConfig.conditionType && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Condition Type
            </Typography>
            <Chip 
              label={conditionConfig.conditionType} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {conditionConfig.expression && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Expression
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
              {conditionConfig.expression}
            </Box>
          </Box>
        )}
        
        {conditionConfig.options && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Typography variant="caption" color="text.secondary" display="block">
              Options
            </Typography>
            
            {conditionConfig.options.map((option: any, index: number) => (
              <Box key={index} sx={{ mt: 0.5 }}>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={option.enabled}
                      disabled
                    />
                  }
                  label={
                    <Typography variant="caption">
                      {option.label}
                    </Typography>
                  }
                />
              </Box>
            ))}
          </Box>
        )}
        
        {conditionConfig.defaultPath && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Default Path
              </Typography>
              <Chip 
                label={conditionConfig.defaultPath} 
                size="small"
                color={conditionConfig.defaultPath === 'true' ? 'success' : 'error'}
                sx={{ fontSize: '0.7rem' }}
              />
            </Box>
          </Box>
        )}
      </Box>
    );
  };
  
  // Inject the condition-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderConditionContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(ConditionNode);
