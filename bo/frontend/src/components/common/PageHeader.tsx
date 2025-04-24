import React, { ReactNode } from 'react';
import { 
  Box, 
  Typography, 
  Breadcrumbs, 
  Link as MuiLink, 
  useTheme 
} from '@mui/material';
import { Link } from 'react-router-dom';

interface BreadcrumbItem {
  label: string;
  path?: string;
}

interface PageHeaderProps {
  title: string;
  breadcrumbs?: BreadcrumbItem[];
  actions?: ReactNode;
  subtitle?: string;
}

/**
 * PageHeader component for consistent page headers across the application.
 * Includes title, optional subtitle, breadcrumbs, and action buttons.
 */
const PageHeader: React.FC<PageHeaderProps> = ({ 
  title, 
  breadcrumbs, 
  actions,
  subtitle
}) => {
  const theme = useTheme();

  return (
    <Box sx={{ mb: 4 }}>
      {/* Breadcrumbs */}
      {breadcrumbs && breadcrumbs.length > 0 && (
        <Breadcrumbs sx={{ mb: 2 }}>
          {breadcrumbs.map((crumb, index) => {
            const isLast = index === breadcrumbs.length - 1;
            
            return isLast ? (
              <Typography color="text.primary" key={index}>
                {crumb.label}
              </Typography>
            ) : (
              <MuiLink 
                component={Link} 
                to={crumb.path || '#'} 
                color="inherit" 
                underline="hover"
                key={index}
              >
                {crumb.label}
              </MuiLink>
            );
          })}
        </Breadcrumbs>
      )}

      {/* Title and Actions */}
      <Box 
        display="flex" 
        justifyContent="space-between" 
        alignItems={subtitle ? 'flex-start' : 'center'}
      >
        <Box>
          <Typography variant="h4" component="h1">
            {title}
          </Typography>
          {subtitle && (
            <Typography 
              variant="subtitle1" 
              color="text.secondary"
              sx={{ mt: 0.5 }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>
        {actions && (
          <Box>
            {actions}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default PageHeader;
