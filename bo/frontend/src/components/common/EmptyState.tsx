import React from 'react';
import { Box, Typography, Button, Paper, SxProps, Theme } from '@mui/material';

interface EmptyStateProps {
  title?: string;
  message: string;
  icon?: React.ReactNode;
  actionLabel?: string;
  onAction?: () => void;
  sx?: SxProps<Theme>;
  paperProps?: React.ComponentProps<typeof Paper>;
}

/**
 * EmptyState component for displaying when no data is available.
 * Can include an icon, message, and action button.
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  message,
  icon,
  actionLabel,
  onAction,
  sx,
  paperProps
}) => {
  return (
    <Paper
      sx={{
        p: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        ...sx
      }}
      {...paperProps}
    >
      {icon && (
        <Box sx={{ mb: 2, color: 'text.secondary' }}>
          {icon}
        </Box>
      )}
      
      {title && (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      )}
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: actionLabel ? 3 : 0 }}>
        {message}
      </Typography>
      
      {actionLabel && onAction && (
        <Button 
          variant="contained" 
          color="primary" 
          onClick={onAction}
        >
          {actionLabel}
        </Button>
      )}
    </Paper>
  );
};

export default EmptyState;
