import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  CircularProgress,
  Grid,
  TextField,
  Typography,
  Divider
} from '@mui/material';
import { NodeData, NodeType, NODE_TYPE_CONFIG } from '../types/nodeTypes';

export interface NodeFormProps {
  open: boolean;
  node: NodeData | null;
  onClose: () => void;
  onSave: (updatedNode: NodeData) => void;
  isLoading?: boolean;
  children?: React.ReactNode;
}

/**
 * Base component for node editing forms
 */
const NodeFormBase: React.FC<NodeFormProps> = ({
  open,
  node,
  onClose,
  onSave,
  isLoading = false,
  children
}) => {
  const [formValues, setFormValues] = React.useState<Partial<NodeData>>({});

  // Initialize form values when node changes
  React.useEffect(() => {
    if (node) {
      setFormValues({
        label: node.label,
        description: node.description || '',
        config: { ...node.config }
      });
    } else {
      setFormValues({});
    }
  }, [node]);

  // Handle form change
  const handleChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle config change
  const handleConfigChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      config: {
        ...(prev.config || {}),
        [name]: value
      }
    }));
  };

  // Handle save
  const handleSave = () => {
    if (!node) return;
    
    const updatedNode: NodeData = {
      ...node,
      label: formValues.label || node.label,
      description: formValues.description,
      config: formValues.config || {}
    };
    
    onSave(updatedNode);
  };

  // Get node type configuration
  const getNodeTypeConfig = () => {
    if (!node) return null;
    return NODE_TYPE_CONFIG[node.type];
  };

  const nodeConfig = getNodeTypeConfig();

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        {node ? `Edit ${nodeConfig?.label} Node` : 'Create Node'}
      </DialogTitle>
      
      <DialogContent>
        <Grid container spacing={2}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              Basic Information
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          {/* Node Label */}
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Node Label"
              name="label"
              value={formValues.label || ''}
              onChange={(e) => handleChange('label', e.target.value)}
              required
            />
          </Grid>
          
          {/* Node Description */}
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={formValues.description || ''}
              onChange={(e) => handleChange('description', e.target.value)}
              multiline
              rows={2}
            />
          </Grid>
          
          {/* Node Type Specific Fields */}
          {children && (
            <>
              <Grid item xs={12} sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Node Configuration
                </Typography>
                <Divider sx={{ mb: 2 }} />
              </Grid>
              
              {/* Render node-specific form fields */}
              {children}
            </>
          )}
        </Grid>
      </DialogContent>
      
      <DialogActions>
        <Button 
          onClick={onClose} 
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button 
          onClick={handleSave} 
          variant="contained" 
          color="primary"
          disabled={isLoading || !formValues.label}
          startIcon={isLoading ? <CircularProgress size={20} /> : undefined}
        >
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default NodeFormBase;
