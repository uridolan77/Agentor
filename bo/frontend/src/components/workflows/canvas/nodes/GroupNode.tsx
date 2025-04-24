import React, { memo, useEffect, useState } from 'react';
import { Box, Typography, Chip, Divider, Badge, alpha } from '@mui/material';
import { Folder as FolderIcon, ExpandMore as ExpandMoreIcon } from '@mui/icons-material';
import BaseNode, { BaseNodeProps } from './BaseNode';
import { NodeData, NodeType } from '../types';
import { useReactFlow, Node } from 'reactflow';

/**
 * Group node component
 * Extends the base node with group-specific functionality and UI
 * Used to group related nodes together
 */
const GroupNode = (props: BaseNodeProps) => {
  const { data, selected } = props;
  const groupConfig = data.config || {};
  const reactFlowInstance = useReactFlow();
  const [childNodeLabels, setChildNodeLabels] = useState<Record<string, string>>({});
  
  // Highlight child nodes when the group is selected
  useEffect(() => {
    if (!groupConfig.childNodes || !Array.isArray(groupConfig.childNodes)) return;
    
    // Get all nodes from ReactFlow
    const allNodes = reactFlowInstance.getNodes();
    
    // Create a map of node IDs to labels
    const nodeLabels: Record<string, string> = {};
    
    // Apply highlighting to child nodes
    if (selected) {
      allNodes.forEach(node => {
        // Store node labels for display
        if (groupConfig.childNodes.includes(node.id)) {
          nodeLabels[node.id] = node.data.label || node.id;
          
          // Apply highlight style to child nodes
          reactFlowInstance.setNodes(nodes => 
            nodes.map(n => {
              if (n.id === node.id) {
                return {
                  ...n,
                  style: {
                    ...n.style,
                    boxShadow: `0 0 8px 2px ${alpha('#607d8b', 0.6)}`,
                    border: '2px dashed #607d8b'
                  }
                };
              }
              return n;
            })
          );
        }
      });
    } else {
      // Remove highlighting when group is not selected
      reactFlowInstance.setNodes(nodes => 
        nodes.map(n => {
          if (groupConfig.childNodes.includes(n.id)) {
            return {
              ...n,
              style: {
                ...n.style,
                boxShadow: 'none',
                border: 'none'
              }
            };
          }
          return n;
        })
      );
    }
    
    setChildNodeLabels(nodeLabels);
  }, [selected, groupConfig.childNodes, reactFlowInstance]);
  
  // Group-specific content to be rendered inside the base node
  const renderGroupContent = () => {
    return (
      <Box className="group-node-content">
        {groupConfig.category && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Category
            </Typography>
            <Chip 
              label={groupConfig.category} 
              size="small" 
              sx={{ mt: 0.5, fontSize: '0.7rem' }}
            />
          </Box>
        )}
        
        {groupConfig.childNodes && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" color="text.secondary" display="block">
              Child Nodes
            </Typography>
            <Badge 
              badgeContent={groupConfig.childNodes.length} 
              color="primary"
              sx={{ mt: 0.5, display: 'block' }}
            >
              <Chip 
                label="Nodes" 
                size="small"
                icon={<FolderIcon fontSize="small" />}
                sx={{ fontSize: '0.7rem' }}
              />
            </Badge>
            
            {/* Display child node labels when selected */}
            {selected && Object.keys(childNodeLabels).length > 0 && (
              <Box sx={{ mt: 1, pl: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  Grouped nodes:
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                  {Object.entries(childNodeLabels).map(([nodeId, label]) => (
                    <Chip
                      key={nodeId}
                      label={label}
                      size="small"
                      variant="outlined"
                      sx={{ 
                        fontSize: '0.65rem', 
                        height: 20,
                        backgroundColor: alpha('#607d8b', 0.1)
                      }}
                    />
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}
        
        {groupConfig.description && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Typography variant="caption" color="text.secondary" display="block">
              Description
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {groupConfig.description}
            </Typography>
          </Box>
        )}
        
        {groupConfig.metadata && (
          <Box sx={{ mt: 1 }}>
            <Divider sx={{ my: 0.5 }} />
            <Typography variant="caption" color="text.secondary" display="block">
              Metadata
            </Typography>
            {Object.entries(groupConfig.metadata).map(([key, value]) => (
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
        
        {/* Expand/Collapse button for the group */}
        <Box sx={{ mt: 1, textAlign: 'center' }}>
          <Divider sx={{ my: 0.5 }} />
          <Chip
            label={groupConfig.expanded ? "Collapse Group" : "Expand Group"}
            size="small"
            icon={<ExpandMoreIcon fontSize="small" />}
            sx={{ fontSize: '0.7rem' }}
            onClick={(e) => {
              e.stopPropagation();
              // This would be handled by the parent component
              console.log('Toggle group expansion');
            }}
          />
        </Box>
      </Box>
    );
  };
  
  // Inject the group-specific content into the base node
  const enhancedData = {
    ...data,
    renderContent: renderGroupContent
  };
  
  return <BaseNode {...props} data={enhancedData} />;
};

export default memo(GroupNode);
