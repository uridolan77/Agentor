import React, { memo } from 'react';
import { EdgeProps, getBezierPath, EdgeLabelRenderer } from 'reactflow';
import { 
  Typography, 
  Tooltip, 
  IconButton, 
  Paper,
  Box 
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Link as LinkIcon,
  LooksOne as LooksOneIcon,
  Filter1 as Filter1Icon,
  FilterNone as FilterNoneIcon
} from '@mui/icons-material';

interface RelationshipData {
  type: 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
  sourceColumn: string;
  targetColumn: string;
}

const RelationshipEdge: React.FC<EdgeProps<RelationshipData>> = ({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // Determine relationship icon based on type
  const getRelationshipIcon = () => {
    switch (data?.type) {
      case 'one-to-one':
        return <LooksOneIcon fontSize="small" />;
      case 'one-to-many':
        return <Filter1Icon fontSize="small" />;
      case 'many-to-one':
        return <Filter1Icon fontSize="small" sx={{ transform: 'rotate(180deg)' }} />;
      case 'many-to-many':
        return <FilterNoneIcon fontSize="small" />;
      default:
        return <LinkIcon fontSize="small" />;
    }
  };

  // Format relationship label text
  const getRelationshipLabel = () => {
    switch (data?.type) {
      case 'one-to-one':
        return '1:1';
      case 'one-to-many':
        return '1:N';
      case 'many-to-one':
        return 'N:1';
      case 'many-to-many':
        return 'N:M';
      default:
        return '';
    }
  };

  return (
    <>
      <path
        id={id}
        style={{
          ...style,
          strokeWidth: 2,
          stroke: '#666',
        }}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      
      <EdgeLabelRenderer>
        <Box
          sx={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 12,
            pointerEvents: 'all',
          }}
        >
          <Paper
            elevation={2}
            sx={{
              padding: '4px 8px',
              borderRadius: 4,
              display: 'flex',
              alignItems: 'center',
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              border: '1px solid #ddd',
            }}
          >
            {getRelationshipIcon()}
            <Typography variant="caption" sx={{ mx: 0.5 }}>
              {getRelationshipLabel()}
            </Typography>
            <Tooltip title="Remove relationship">
              <IconButton
                size="small"
                onClick={(event) => {
                  event.stopPropagation();
                  console.log('Remove edge', id);
                  // Here you can dispatch an action to remove the edge
                }}
                sx={{ ml: 0.5, p: 0.5 }}
              >
                <DeleteIcon fontSize="small" color="error" />
              </IconButton>
            </Tooltip>
          </Paper>
          
          <Tooltip
            title={
              <span>
                {source}.{data?.sourceColumn} → {target}.{data?.targetColumn}
              </span>
            }
          >
            <Box
              sx={{
                position: 'absolute',
                top: '100%',
                left: '50%',
                transform: 'translateX(-50%)',
                mt: 0.5,
                fontSize: 10,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                padding: '2px 4px',
                borderRadius: 1,
                whiteSpace: 'nowrap',
              }}
            >
              {data?.sourceColumn} → {data?.targetColumn}
            </Box>
          </Tooltip>
        </Box>
      </EdgeLabelRenderer>
    </>
  );
};

export default memo(RelationshipEdge);