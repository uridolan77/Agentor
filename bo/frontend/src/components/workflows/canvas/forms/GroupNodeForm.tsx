import React from 'react';
import {
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Switch,
  FormControlLabel,
  Chip,
  Box,
  Typography,
  Divider
} from '@mui/material';
import NodeFormBase, { NodeFormProps } from './NodeFormBase';
import { NodeData, NodeType } from '../types/nodeTypes';
import { Node } from 'reactflow';

interface GroupNodeFormProps extends Omit<NodeFormProps, 'children'> {
  availableNodes?: Node<NodeData>[];
}

/**
 * Form for editing Group nodes
 */
const GroupNodeForm: React.FC<GroupNodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading,
  availableNodes = []
}) => {
  const [formValues, setFormValues] = React.useState<{
    childNodes?: string[];
    collapsed?: boolean;
    backgroundColor?: string;
    borderColor?: string;
  }>({
    childNodes: [],
    collapsed: false,
    backgroundColor: 'rgba(240, 240, 240, 0.8)',
    borderColor: '#ccc'
  });

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        childNodes: node.config.childNodes || [],
        collapsed: node.config.collapsed || false,
        backgroundColor: node.config.backgroundColor || 'rgba(240, 240, 240, 0.8)',
        borderColor: node.config.borderColor || '#ccc'
      });
    } else {
      setFormValues({
        childNodes: [],
        collapsed: false,
        backgroundColor: 'rgba(240, 240, 240, 0.8)',
        borderColor: '#ccc'
      });
    }
  }, [node]);

  // Handle form change
  const handleChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle child nodes change
  const handleChildNodesChange = (nodeIds: string[]) => {
    handleChange('childNodes', nodeIds);
  };

  // Handle save
  const handleSave = (updatedNode: NodeData) => {
    const nodeWithConfig: NodeData = {
      ...updatedNode,
      config: {
        ...updatedNode.config,
        childNodes: formValues.childNodes,
        collapsed: formValues.collapsed,
        backgroundColor: formValues.backgroundColor,
        borderColor: formValues.borderColor
      }
    };
    
    onSave(nodeWithConfig);
  };

  // Get available nodes for grouping (excluding the current node and already grouped nodes)
  const getAvailableNodesForGrouping = () => {
    if (!availableNodes) return [];
    
    return availableNodes.filter(availableNode => {
      // Exclude the current node
      if (node && availableNode.id === node.id) return false;
      
      // Exclude nodes that are already in a group
      if (availableNode.data.parentId) return false;
      
      // Exclude nodes that are already in this group
      if (formValues.childNodes?.includes(availableNode.id)) return false;
      
      return true;
    });
  };

  // Get nodes that are already in this group
  const getGroupedNodes = () => {
    if (!availableNodes || !formValues.childNodes) return [];
    
    return availableNodes.filter(availableNode => 
      formValues.childNodes?.includes(availableNode.id)
    );
  };

  const availableNodesForGrouping = getAvailableNodesForGrouping();
  const groupedNodes = getGroupedNodes();

  return (
    <NodeFormBase
      open={open}
      node={node}
      onClose={onClose}
      onSave={handleSave}
      isLoading={isLoading}
    >
      {/* Group Appearance */}
      <Grid item xs={12}>
        <Typography variant="subtitle2" gutterBottom>
          Group Appearance
        </Typography>
        <Divider sx={{ mb: 2 }} />
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Background Color"
          name="backgroundColor"
          value={formValues.backgroundColor || 'rgba(240, 240, 240, 0.8)'}
          onChange={(e) => handleChange('backgroundColor', e.target.value)}
          helperText="CSS color or rgba value"
        />
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Border Color"
          name="borderColor"
          value={formValues.borderColor || '#ccc'}
          onChange={(e) => handleChange('borderColor', e.target.value)}
          helperText="CSS color value"
        />
      </Grid>
      
      <Grid item xs={12}>
        <FormControlLabel
          control={
            <Switch
              checked={formValues.collapsed || false}
              onChange={(e) => handleChange('collapsed', e.target.checked)}
              name="collapsed"
            />
          }
          label="Collapsed by default"
        />
      </Grid>
      
      {/* Group Nodes */}
      <Grid item xs={12}>
        <Typography variant="subtitle2" gutterBottom>
          Grouped Nodes
        </Typography>
        <Divider sx={{ mb: 2 }} />
      </Grid>
      
      {/* Currently grouped nodes */}
      <Grid item xs={12}>
        <Typography variant="body2" gutterBottom>
          Nodes in this group:
        </Typography>
        
        {groupedNodes.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
            No nodes in this group yet
          </Typography>
        ) : (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
            {groupedNodes.map(groupedNode => (
              <Chip 
                key={groupedNode.id} 
                label={groupedNode.data.label} 
                onDelete={() => {
                  const newChildNodes = formValues.childNodes?.filter(id => id !== groupedNode.id) || [];
                  handleChildNodesChange(newChildNodes);
                }}
              />
            ))}
          </Box>
        )}
      </Grid>
      
      {/* Add nodes to group */}
      <Grid item xs={12}>
        <FormControl fullWidth>
          <InputLabel id="add-node-select-label">Add Node to Group</InputLabel>
          <Select
            labelId="add-node-select-label"
            id="add-node-select"
            value=""
            label="Add Node to Group"
            onChange={(e) => {
              const nodeId = e.target.value as string;
              if (nodeId) {
                const newChildNodes = [...(formValues.childNodes || []), nodeId];
                handleChildNodesChange(newChildNodes);
              }
            }}
            disabled={availableNodesForGrouping.length === 0}
          >
            <MenuItem value="">
              <em>Select a node to add</em>
            </MenuItem>
            
            {availableNodesForGrouping.map(availableNode => (
              <MenuItem key={availableNode.id} value={availableNode.id}>
                {availableNode.data.label} ({availableNode.type})
              </MenuItem>
            ))}
          </Select>
          <FormHelperText>
            {availableNodesForGrouping.length === 0 
              ? 'No available nodes to add to this group' 
              : 'Select a node to add to this group'}
          </FormHelperText>
        </FormControl>
      </Grid>
    </NodeFormBase>
  );
};

export default GroupNodeForm;
