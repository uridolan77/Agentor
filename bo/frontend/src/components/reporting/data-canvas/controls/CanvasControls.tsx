import React from 'react';
import { Paper, Box, IconButton, Tooltip, Divider } from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  GridOn as GridOnIcon,
  GridOff as GridOffIcon
} from '@mui/icons-material';

interface CanvasControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
  onSave?: () => void;
  onDelete?: () => void;
  onToggleGrid?: () => void;
  showGrid?: boolean;
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
  onSave,
  onDelete,
  onToggleGrid,
  showGrid = true,
  hasSelection = false
}) => {
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
      
      {/* Grid Controls */}
      {onToggleGrid && (
        <Box>
          <Tooltip title={showGrid ? "Hide Grid" : "Show Grid"} placement="left">
            <IconButton size="small" onClick={onToggleGrid}>
              {showGrid ? <GridOffIcon fontSize="small" /> : <GridOnIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
      )}
      
      {/* Action Controls */}
      {(onDelete || onSave) && (
        <>
          <Divider sx={{ my: 0.5 }} />
          <Box>
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
            
            {onSave && (
              <Tooltip title="Save Model" placement="left">
                <IconButton size="small" onClick={onSave} color="primary">
                  <SaveIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </>
      )}
    </Paper>
  );
};

export default CanvasControls;
