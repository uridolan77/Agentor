import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';

export type StatusType = 'active' | 'inactive' | 'pending' | 'error' | 'warning' | 'success';

interface StatusBadgeProps {
  status: StatusType | string;
  size?: 'small' | 'medium' | 'large';
  withLabel?: boolean;
}

/**
 * StatusBadge component for displaying status indicators.
 * Used to visually represent the status of agents, tools, workflows, etc.
 */
const StatusBadge: React.FC<StatusBadgeProps> = ({ 
  status, 
  size = 'medium',
  withLabel = true
}) => {
  const theme = useTheme();
  
  // Define status colors and labels
  const statusConfig: Record<string, { color: string; label: string; textColor: string }> = {
    active: { 
      color: theme.palette.success.main, 
      label: 'Active',
      textColor: theme.palette.success.contrastText
    },
    inactive: { 
      color: theme.palette.text.disabled, 
      label: 'Inactive',
      textColor: theme.palette.background.paper
    },
    pending: { 
      color: theme.palette.warning.main, 
      label: 'Pending',
      textColor: theme.palette.warning.contrastText
    },
    error: { 
      color: theme.palette.error.main, 
      label: 'Error',
      textColor: theme.palette.error.contrastText
    },
    warning: { 
      color: theme.palette.warning.main, 
      label: 'Warning',
      textColor: theme.palette.warning.contrastText
    },
    success: { 
      color: theme.palette.success.main, 
      label: 'Success',
      textColor: theme.palette.success.contrastText
    }
  };

  // Default to inactive if status is not recognized
  const config = statusConfig[status.toLowerCase()] || statusConfig.inactive;
  
  // Size configurations
  const sizeConfig = {
    small: {
      px: 1,
      py: 0.25,
      fontSize: '0.75rem',
      borderRadius: 1
    },
    medium: {
      px: 1.5,
      py: 0.5,
      fontSize: '0.875rem',
      borderRadius: 1
    },
    large: {
      px: 2,
      py: 0.75,
      fontSize: '1rem',
      borderRadius: 1
    }
  };

  return (
    <Box
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        bgcolor: config.color,
        color: config.textColor,
        ...sizeConfig[size],
        fontWeight: 'medium'
      }}
    >
      {withLabel ? (
        <Typography 
          variant="caption" 
          component="span"
          sx={{ 
            fontSize: sizeConfig[size].fontSize,
            fontWeight: 'medium'
          }}
        >
          {config.label}
        </Typography>
      ) : (
        <Box 
          sx={{ 
            width: sizeConfig[size].fontSize, 
            height: sizeConfig[size].fontSize,
            borderRadius: '50%',
            bgcolor: 'currentColor'
          }} 
        />
      )}
    </Box>
  );
};

export default StatusBadge;
