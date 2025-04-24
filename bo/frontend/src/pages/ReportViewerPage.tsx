import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  Grid,
  Skeleton,
  Divider,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Breadcrumbs,
  Link,
  CircularProgress,
  Alert,
  Snackbar,
  Stack,
  Chip
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  MoreVert as MoreIcon,
  Share as ShareIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
  Dashboard as DashboardIcon,
  Print as PrintIcon,
  NavigateNext as NavigateNextIcon
} from '@mui/icons-material';
import useReportStore from '../store/reportStore';
import { ReportExecutionResult } from '../types/reporting';
import { ChartData, VisualizationConfig } from '../types/reporting/reportTypes';

// Import chart components
// These would be your visualization components, we'll add placeholders for now
const TableChart = React.lazy(() => import('../components/charts/TableChart'));
const BarChart = React.lazy(() => import('../components/charts/BarChart'));
const LineChart = React.lazy(() => import('../components/charts/LineChart'));
const PieChart = React.lazy(() => import('../components/charts/PieChart'));
const MetricDisplay = React.lazy(() => import('../components/charts/MetricDisplay'));

const ReportViewerPage: React.FC = () => {
  const { reportId } = useParams<{ reportId: string }>();
  const navigate = useNavigate();

  const {
    selectedReport,
    fetchReportById,
    executeReport,
    reportExecutionResult,
    executingReport,
    dataSources,
    fetchDataSources
  } = useReportStore();

  const [error, setError] = useState<string | null>(null);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [dataSource, setDataSource] = useState<string>('');

  // Notification state
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Load report and execute it on component mount
  useEffect(() => {
    const loadReportData = async () => {
      if (!reportId) return;

      try {
        setError(null);

        // Load data sources if needed
        if (dataSources.length === 0) {
          await fetchDataSources();
        }

        // Load report details
        await fetchReportById(reportId);

        // Execute the report
        await executeReport(reportId);
      } catch (err) {
        console.error('Error loading report:', err);
        setError('Failed to load report data. Please try again later.');
      }
    };

    loadReportData();
  }, [reportId, fetchReportById, executeReport, fetchDataSources, dataSources.length]);

  // Update data source name when selectedReport changes
  useEffect(() => {
    if (selectedReport && dataSources.length > 0) {
      const source = dataSources.find(ds => ds.id === selectedReport.dataSourceId);
      setDataSource(source ? source.name : 'Unknown Data Source');
    }
  }, [selectedReport, dataSources]);

  // Handle menu open
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  // Handle edit report
  const handleEditReport = () => {
    if (reportId) {
      navigate(`/reports/builder?id=${reportId}`);
    }
  };

  // Handle refresh report
  const handleRefreshReport = async () => {
    if (!reportId) return;

    try {
      setRefreshing(true);
      await executeReport(reportId);
      setNotification({
        open: true,
        message: 'Report refreshed successfully',
        severity: 'success'
      });
    } catch (err) {
      console.error('Error refreshing report:', err);
      setNotification({
        open: true,
        message: 'Failed to refresh report data',
        severity: 'error'
      });
    } finally {
      setRefreshing(false);
    }
  };

  // Handle toggle favorite
  const handleToggleFavorite = () => {
    setNotification({
      open: true,
      message: 'Favorite toggling feature coming soon',
      severity: 'info'
    });
  };

  // Handle share report
  const handleShareReport = () => {
    handleMenuClose();
    setNotification({
      open: true,
      message: 'Sharing feature coming soon',
      severity: 'info'
    });
  };

  // Handle print report
  const handlePrintReport = () => {
    handleMenuClose();
    window.print();
  };

  // Handle export to CSV/Excel
  const handleExportData = (format: 'csv' | 'excel') => {
    handleMenuClose();
    setNotification({
      open: true,
      message: `Export to ${format.toUpperCase()} feature coming soon`,
      severity: 'info'
    });
  };

  // Helper function to render visualization
  const renderVisualization = (
    visualizationType: string,
    visualizationId: string,
    dimensions: string[],
    metrics: string[],
    settings: Record<string, any>
  ) => {
    if (!reportExecutionResult || !reportExecutionResult.data) {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            No data available
          </Typography>
        </Box>
      );
    }

    const { columns, rows } = reportExecutionResult.data;

    // Render different chart types based on the visualization type
    switch (visualizationType) {
      case 'table':
        return (
          <React.Suspense fallback={<Skeleton variant="rectangular" height={300} />}>
            <TableChart data={rows as ChartData[]} config={settings as VisualizationConfig} />
          </React.Suspense>
        );

      case 'bar':
        return (
          <React.Suspense fallback={<Skeleton variant="rectangular" height={300} />}>
            <BarChart data={rows as ChartData[]} config={settings as VisualizationConfig} />
          </React.Suspense>
        );

      case 'line':
        return (
          <React.Suspense fallback={<Skeleton variant="rectangular" height={300} />}>
            <LineChart data={rows as ChartData[]} config={settings as VisualizationConfig} />
          </React.Suspense>
        );

      case 'pie':
        return (
          <React.Suspense fallback={<Skeleton variant="rectangular" height={300} />}>
            <PieChart data={rows as ChartData[]} config={settings as VisualizationConfig} />
          </React.Suspense>
        );

      case 'metric':
        return (
          <React.Suspense fallback={<Skeleton variant="rectangular" height={300} />}>
            <MetricDisplay data={rows as ChartData[]} config={settings as VisualizationConfig} />
          </React.Suspense>
        );

      default:
        return (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary">
              Unsupported visualization type: {visualizationType}
            </Typography>
          </Box>
        );
    }
  };

  // Function to render loading state
  const renderLoadingState = () => (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          <Skeleton width={200} />
        </Typography>
      </Box>

      <Paper sx={{ p: 3, mb: 4 }}>
        <Skeleton variant="text" sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={400} sx={{ mb: 2 }} />
        <Skeleton variant="text" width="60%" />
      </Paper>
    </Container>
  );

  // Function to render error state
  const renderErrorState = () => (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>

      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Something went wrong
        </Typography>
        <Typography variant="body1" paragraph>
          We couldn't load the report data. Please try again or contact support if the problem persists.
        </Typography>
        <Button
          variant="contained"
          onClick={() => navigate('/reports')}
        >
          Back to Reports
        </Button>
      </Paper>
    </Container>
  );

  // If there's an error, show the error state
  if (error) {
    return renderErrorState();
  }

  // If we're loading the report or executing it, show loading state
  if (!selectedReport || (!reportExecutionResult && executingReport)) {
    return renderLoadingState();
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4, pb: 6 }}>
      {/* Breadcrumbs navigation */}
      <Breadcrumbs
        separator={<NavigateNextIcon fontSize="small" />}
        aria-label="breadcrumb"
        sx={{ mb: 2 }}
      >
        <Link color="inherit" onClick={() => navigate('/dashboard')} sx={{ cursor: 'pointer' }}>
          Dashboard
        </Link>
        <Link color="inherit" onClick={() => navigate('/reports')} sx={{ cursor: 'pointer' }}>
          Reports
        </Link>
        <Typography color="text.primary">
          {selectedReport?.name || 'Report Details'}
        </Typography>
      </Breadcrumbs>

      {/* Report Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            {selectedReport.name}
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" label={dataSource} />
            <Typography variant="body2" color="text.secondary">
              Last updated: {new Date(selectedReport.updatedAt).toLocaleString()}
            </Typography>
            {selectedReport.lastRunAt && (
              <Typography variant="body2" color="text.secondary">
                Last run: {new Date(selectedReport.lastRunAt).toLocaleString()}
              </Typography>
            )}
          </Stack>
        </Box>

        <Box>
          <Tooltip title={selectedReport.isFavorite ? "Remove from favorites" : "Add to favorites"}>
            <IconButton
              color={selectedReport.isFavorite ? "secondary" : "default"}
              onClick={handleToggleFavorite}
            >
              {selectedReport.isFavorite ? <FavoriteIcon /> : <FavoriteBorderIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh data">
            <IconButton
              onClick={handleRefreshReport}
              disabled={refreshing || executingReport}
            >
              {(refreshing || executingReport) ? <CircularProgress size={24} /> : <RefreshIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Edit report">
            <IconButton onClick={handleEditReport}>
              <EditIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="More options">
            <IconButton
              aria-controls="report-menu"
              aria-haspopup="true"
              onClick={handleMenuOpen}
            >
              <MoreIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Report description */}
      {selectedReport.description && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="body1">{selectedReport.description}</Typography>
        </Paper>
      )}

      {/* Report execution error */}
      {reportExecutionResult?.status === 'error' && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {reportExecutionResult.error || 'An error occurred while executing the report'}
        </Alert>
      )}

      {/* Report Visualizations */}
      {executingReport ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 6 }}>
          <CircularProgress />
          <Typography variant="body1" color="text.secondary" sx={{ ml: 2 }}>
            Executing report...
          </Typography>
        </Box>
      ) : (
        <>
          {selectedReport.configuration?.visualizations?.length > 0 ? (
            <Grid container spacing={3}>
              {selectedReport.configuration.visualizations.map((visualization: any) => (
                <Grid item xs={12} md={visualization.type === 'metric' ? 3 : 12} key={visualization.id}>
                  <Paper sx={{ p: 3, height: '100%' }}>
                    <Typography variant="h6" gutterBottom>
                      {visualization.title}
                    </Typography>

                    <Divider sx={{ mb: 2 }} />

                    <Box sx={{ minHeight: visualization.type === 'metric' ? 120 : 300 }}>
                      {renderVisualization(
                        visualization.type,
                        visualization.id,
                        visualization.dimensions,
                        visualization.metrics,
                        visualization.settings
                      )}
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                No visualizations defined
              </Typography>
              <Typography variant="body1" gutterBottom>
                This report doesn't have any visualizations configured.
              </Typography>
              <Button
                variant="contained"
                startIcon={<EditIcon />}
                onClick={handleEditReport}
                sx={{ mt: 2 }}
              >
                Edit Report
              </Button>
            </Paper>
          )}
        </>
      )}

      {/* Actions Menu */}
      <Menu
        id="report-menu"
        anchorEl={menuAnchorEl}
        keepMounted
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleShareReport}>
          <ListItemIcon>
            <ShareIcon fontSize="small" />
          </ListItemIcon>
          Share Report
        </MenuItem>
        <MenuItem onClick={handlePrintReport}>
          <ListItemIcon>
            <PrintIcon fontSize="small" />
          </ListItemIcon>
          Print
        </MenuItem>
        <MenuItem onClick={() => handleExportData('csv')}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          Export as CSV
        </MenuItem>
        <MenuItem onClick={() => handleExportData('excel')}>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          Export as Excel
        </MenuItem>
      </Menu>

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification(prev => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setNotification(prev => ({ ...prev, open: false }))}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

type ListItemIconProps = {
  children: React.ReactNode;
  fontSize?: 'small' | 'medium' | 'large';
  sx?: any;
};

const ListItemIcon: React.FC<ListItemIconProps> = ({ children, fontSize, sx }) => (
  <Box sx={{ mr: 2, minWidth: '24px', display: 'flex', ...sx }}>
    {children}
  </Box>
);

export default ReportViewerPage;
