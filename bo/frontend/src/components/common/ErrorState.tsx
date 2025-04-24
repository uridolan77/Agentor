import React from 'react';
import { Box, Typography, Button, Paper, SxProps, Theme } from '@mui/material';
import { Error as ErrorIcon } from '@mui/icons-material';

interface ErrorStateProps {
  title?: string;
  message: string;
  retryLabel?: string;
  onRetry?: () => void;
  error?: Error | string;
  showDetails?: boolean;
  sx?: SxProps<Theme>;
  paperProps?: React.ComponentProps<typeof Paper>;
}

/**
 * ErrorState component for displaying error messages.
 * Includes options for retry action and displaying error details.
 */
const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'An error occurred',
  message,
  retryLabel = 'Retry',
  onRetry,
  error,
  showDetails = false,
  sx,
  paperProps
}) => {
  // Format error details
  const errorDetails = React.useMemo(() => {
    if (!error) return null;
    
    if (typeof error === 'string') {
      return error;
    }
    
    return error.message || 'Unknown error';
  }, [error]);

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
      <ErrorIcon color="error" sx={{ fontSize: 48, mb: 2 }} />
      
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        {message}
      </Typography>
      
      {showDetails && errorDetails && (
        <Box 
          sx={{ 
            bgcolor: 'error.light', 
            color: 'error.contrastText',
            p: 2, 
            borderRadius: 1, 
            width: '100%', 
            mb: 3,
            maxHeight: '150px',
            overflow: 'auto',
            textAlign: 'left'
          }}
        >
          <Typography variant="body2" component="pre" sx={{ m: 0, fontFamily: 'monospace' }}>
            {errorDetails}
          </Typography>
        </Box>
      )}
      
      {onRetry && (
        <Button 
          variant="contained" 
          color="primary" 
          onClick={onRetry}
        >
          {retryLabel}
        </Button>
      )}
    </Paper>
  );
};

export default ErrorState;
