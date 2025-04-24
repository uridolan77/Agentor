import React from 'react';
import { Box, CircularProgress, Typography, SxProps, Theme } from '@mui/material';

interface LoadingStateProps {
  message?: string;
  size?: number;
  fullHeight?: boolean;
  sx?: SxProps<Theme>;
}

/**
 * LoadingState component for displaying a loading indicator.
 * Can be used as a full-page loader or inline.
 */
const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading...',
  size = 40,
  fullHeight = false,
  sx
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 3,
        height: fullHeight ? '100%' : 'auto',
        minHeight: fullHeight ? '200px' : 'auto',
        ...sx
      }}
    >
      <CircularProgress size={size} />
      {message && (
        <Typography 
          variant="body2" 
          color="text.secondary" 
          sx={{ mt: 2 }}
        >
          {message}
        </Typography>
      )}
    </Box>
  );
};

export default LoadingState;
