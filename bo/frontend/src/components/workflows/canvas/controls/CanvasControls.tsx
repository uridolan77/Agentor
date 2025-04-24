import React from 'react';
import { Paper, Box, IconButton, Tooltip, Divider, useTheme } from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  GridOn as GridOnIcon,
  GridOff as GridOffIcon
} from '@mui/icons-material';

interface CanvasControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onSave?: () => void;
  onDelete?: () => void;
  onToggleGrid?: () => void;
  showGrid?: boolean;
  canUndo?: boolean;
  canRedo?: boolean;
  hasSelection?: boolean;
}

/**
 * CanvasControls component
 * Provides controls for zooming, panning, and other canvas operations
 */
const CanvasControls: React.FC<CanvasControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onFitView,
  onUndo,
  onRedo,
  onSave,
  onDelete,
  onToggleGrid,
  showGrid = true,
  canUndo = false,
  canRedo = false,
  hasSelection = false
}) => {
  const theme = useTheme();

  return (
    <Paper
      elevation={2}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        p: 0.5,
        borderRadius: 1,
        bgcolor: 'background.paper'
      }}
    >
      {/* Zoom Controls */}
      <Box>
        <Tooltip title="Zoom In" placement="left">
          <IconButton size="small" onClick={onZoomIn}>
            <ZoomInIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Zoom Out" placement="left">
          <IconButton size="small" onClick={onZoomOut}>
            <ZoomOutIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Fit View" placement="left">
          <IconButton size="small" onClick={onFitView}>
            <CenterIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      <Divider sx={{ my: 0.5 }} />
      
      {/* History Controls */}
      {(onUndo || onRedo) && (
        <Box>
          {onUndo && (
            <Tooltip title="Undo" placement="left">
              <span>
                <IconButton 
                  size="small" 
                  onClick={onUndo} 
                  disabled={!canUndo}
                >
                  <UndoIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          )}
          
          {onRedo && (
            <Tooltip title="Redo" placement="left">
              <span>
                <IconButton 
                  size="small" 
                  onClick={onRedo} 
                  disabled={!canRedo}
                >
                  <RedoIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          )}
        </Box>
      )}
      
      {(onUndo || onRedo) && <Divider sx={{ my: 0.5 }} />}
      
      {/* Action Controls */}
      <Box>
        {onSave && (
          <Tooltip title="Save Workflow" placement="left">
            <IconButton 
              size="small" 
              onClick={onSave}
              color="primary"
            >
              <SaveIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        
        {onDelete && (
          <Tooltip title="Delete Selected" placement="left">
            <span>
              <IconButton 
                size="small" 
                onClick={onDelete} 
                disabled={!hasSelection}
                color="error"
              >
                <DeleteIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        )}
        
        {onToggleGrid && (
          <Tooltip title={showGrid ? "Hide Grid" : "Show Grid"} placement="left">
            <IconButton size="small" onClick={onToggleGrid}>
              {showGrid ? <GridOffIcon fontSize="small" /> : <GridOnIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </Paper>
  );
};

export default CanvasControls;
