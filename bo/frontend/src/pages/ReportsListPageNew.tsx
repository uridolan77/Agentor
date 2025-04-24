import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  IconButton,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  Divider,
  Chip,
  Tooltip,
  CircularProgress,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import {
  Search as SearchIcon,
  Add as AddIcon,
  MoreVert as MoreIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
  ContentCopy as CloneIcon,
  GetApp as DownloadIcon,
  Share as ShareIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  Public as PublicIcon,
  Lock as PrivateIcon,
} from '@mui/icons-material';

import { useReportsListStore, useDataSourcesStore } from '../store/reporting/reportStore';
import { Report, DataSource } from '../types/reporting/reportTypes';

// Styled components
const ReportsContainer = styled(Container)(({ theme }) => ({
  padding: theme.spacing(3),
}));

const PageHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing(3),
}));

const SearchBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(3),
}));

const ReportCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[4],
  },
}));

const ReportCardContent = styled(CardContent)(({ theme }) => ({
  flexGrow: 1,
}));

const ReportTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
  marginBottom: theme.spacing(1),
  display: '-webkit-box',
  WebkitLineClamp: 2,
  WebkitBoxOrient: 'vertical',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}));

const ReportDescription = styled(Typography)(({ theme }) => ({
  color: theme.palette.text.secondary,
  marginBottom: theme.spacing(1),
  display: '-webkit-box',
  WebkitLineClamp: 3,
  WebkitBoxOrient: 'vertical',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}));

const ReportMeta = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginTop: theme.spacing(1),
}));

const ReportDate = styled(Typography)(({ theme }) => ({
  color: theme.palette.text.secondary,
  fontSize: '0.75rem',
}));

const ReportCardActions = styled(CardActions)(({ theme }) => ({
  justifyContent: 'space-between',
  padding: theme.spacing(1, 2),
}));

const LoadingContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '50vh',
}));

const NoReportsContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '50vh',
  textAlign: 'center',
}));

const ChipsContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(1),
  flexWrap: 'wrap',
  marginTop: theme.spacing(1),
}));

interface ReportsListPageProps {
  onCreateReport?: () => void;
  onViewReport?: (reportId: string) => void;
  onEditReport?: (reportId: string) => void;
}

const ReportsListPageNew: React.FC<ReportsListPageProps> = ({
  onCreateReport,
  onViewReport,
  onEditReport,
}) => {
  const navigate = useNavigate();
  const { reports, isLoading, error, loadReports, deleteReport, toggleFavorite } = useReportsListStore();
  const { dataSources, loadDataSources } = useDataSourcesStore();

  const [searchQuery, setSearchQuery] = useState<string>('');
  const [menuAnchor, setMenuAnchor] = useState<{ el: HTMLElement; reportId: string } | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [reportToDelete, setReportToDelete] = useState<Report | null>(null);
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null);

  // Load reports and data sources
  useEffect(() => {
    loadReports();
    loadDataSources();
  }, []);

  // Handle search
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // Handle menu open/close
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, reportId: string) => {
    event.stopPropagation();
    setMenuAnchor({ el: event.currentTarget, reportId });
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  // Handle favorite toggle
  const handleFavoriteToggle = (event: React.MouseEvent<HTMLElement>, report: Report) => {
    event.stopPropagation();
    toggleFavorite(report.id);
  };

  // Handle report actions
  const handleViewReport = (reportId: string) => {
    if (onViewReport) {
      onViewReport(reportId);
    } else {
      navigate(`/reports/${reportId}`);
    }
  };

  const handleEditReport = (reportId: string) => {
    if (onEditReport) {
      onEditReport(reportId);
    } else {
      navigate(`/reports/${reportId}/edit`);
    }
  };

  const handleCreateReport = () => {
    if (onCreateReport) {
      onCreateReport();
    } else {
      navigate('/reports/new');
    }
  };

  // Handle delete dialog
  const handleDeleteClick = (report: Report) => {
    setReportToDelete(report);
    setDeleteDialogOpen(true);
    handleMenuClose();
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setReportToDelete(null);
  };

  const handleDeleteConfirm = async () => {
    if (!reportToDelete) return;

    try {
      await deleteReport(reportToDelete.id);
      setNotification({
        message: `Report "${reportToDelete.name}" deleted successfully`,
        type: 'success'
      });
    } catch (error) {
      setNotification({
        message: `Failed to delete report: ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error'
      });
    } finally {
      setDeleteDialogOpen(false);
      setReportToDelete(null);
    }
  };

  // Handle notification close
  const handleNotificationClose = () => {
    setNotification(null);
  };

  // Show notification
  const showNotification = (message: string, type: 'success' | 'error' | 'info') => {
    setNotification({ message, type });
  };

  // Get data source name
  const getDataSourceName = (dataSourceId: string) => {
    const dataSource = dataSources.find(ds => ds.id === dataSourceId);
    return dataSource ? dataSource.name : 'Unknown';
  };

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  // Filter reports based on search query
  const filteredReports = reports.filter((report: Report) =>
    report.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (report.description && report.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  // Render loading state
  if (isLoading && reports.length === 0) {
    return (
      <LoadingContainer>
        <CircularProgress size={40} />
        <Typography variant="body1" color="textSecondary" sx={{ mt: 2 }}>
          Loading reports...
        </Typography>
      </LoadingContainer>
    );
  }

  return (
    <ReportsContainer maxWidth="xl">
      {/* Page Header */}
      <PageHeader>
        <Typography variant="h4" component="h1">
          Reports
        </Typography>

        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={handleCreateReport}
        >
          Create Report
        </Button>
      </PageHeader>

      {/* Search Box */}
      <SearchBox>
        <TextField
          fullWidth
          placeholder="Search reports..."
          value={searchQuery}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          variant="outlined"
          size="small"
        />
      </SearchBox>

      {/* Reports Grid */}
      {filteredReports.length === 0 ? (
        <NoReportsContainer>
          <Typography variant="h6" color="textSecondary">
            No reports found
          </Typography>
          <Typography variant="body1" color="textSecondary" sx={{ mt: 1 }}>
            {searchQuery ? 'Try a different search term' : 'Create your first report to get started'}
          </Typography>
          {!searchQuery && (
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={handleCreateReport}
              sx={{ mt: 2 }}
            >
              Create Report
            </Button>
          )}
        </NoReportsContainer>
      ) : (
        <Grid container spacing={3}>
          {filteredReports.map((report: Report) => (
            <Grid item xs={12} sm={6} md={4} key={report.id}>
              <ReportCard>
                <ReportCardContent onClick={() => handleViewReport(report.id)}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <ReportTitle variant="h6">
                      {report.name}
                    </ReportTitle>
                    <Tooltip title={report.isPublic ? 'Public' : 'Private'}>
                      <Box component="span">
                        {report.isPublic ? <PublicIcon fontSize="small" color="action" /> : <PrivateIcon fontSize="small" color="action" />}
                      </Box>
                    </Tooltip>
                  </Box>

                  {report.description && (
                    <ReportDescription variant="body2">
                      {report.description}
                    </ReportDescription>
                  )}

                  <ChipsContainer>
                    <Chip
                      size="small"
                      label={getDataSourceName(report.dataSourceId)}
                      color="primary"
                      variant="outlined"
                    />
                    {report.lastRunAt && (
                      <Chip
                        size="small"
                        label={`Last run: ${formatDate(report.lastRunAt)}`}
                        variant="outlined"
                      />
                    )}
                  </ChipsContainer>

                  <ReportMeta>
                    <ReportDate>
                      Created: {formatDate(report.createdAt)}
                    </ReportDate>
                  </ReportMeta>
                </ReportCardContent>

                <ReportCardActions>
                  <Box>
                    <IconButton
                      size="small"
                      onClick={(e) => handleFavoriteToggle(e, report)}
                      color={report.isFavorite ? 'primary' : 'default'}
                    >
                      {report.isFavorite ? <FavoriteIcon /> : <FavoriteBorderIcon />}
                    </IconButton>

                    <IconButton
                      size="small"
                      onClick={() => handleViewReport(report.id)}
                    >
                      <ViewIcon />
                    </IconButton>

                    <IconButton
                      size="small"
                      onClick={() => handleEditReport(report.id)}
                    >
                      <EditIcon />
                    </IconButton>
                  </Box>

                  <IconButton
                    size="small"
                    onClick={(e) => handleMenuOpen(e, report.id)}
                  >
                    <MoreIcon />
                  </IconButton>
                </ReportCardActions>
              </ReportCard>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Report Menu */}
      <Menu
        anchorEl={menuAnchor?.el}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => {
          if (menuAnchor) {
            const report = reports.find((r: Report) => r.id === menuAnchor.reportId);
            if (report) {
              // Clone report functionality would go here
              showNotification(`Duplicating "${report.name}"...`, 'info');
              handleMenuClose();
            }
          }
        }}>
          <CloneIcon fontSize="small" sx={{ mr: 1 }} />
          Duplicate
        </MenuItem>

        <MenuItem onClick={() => {
          if (menuAnchor) {
            const report = reports.find((r: Report) => r.id === menuAnchor.reportId);
            if (report) {
              // Download/export functionality would go here
              showNotification(`Exporting "${report.name}"...`, 'info');
              handleMenuClose();
            }
          }
        }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
          Export
        </MenuItem>

        <MenuItem onClick={() => {
          if (menuAnchor) {
            const report = reports.find((r: Report) => r.id === menuAnchor.reportId);
            if (report) {
              // Share functionality would go here
              showNotification(`Sharing options for "${report.name}"...`, 'info');
              handleMenuClose();
            }
          }
        }}>
          <ShareIcon fontSize="small" sx={{ mr: 1 }} />
          Share
        </MenuItem>

        <Divider />

        <MenuItem
          sx={{ color: 'error.main' }}
          onClick={() => {
            if (menuAnchor) {
              const report = reports.find((r: Report) => r.id === menuAnchor.reportId);
              if (report) {
                handleDeleteClick(report);
              }
            }
          }}
        >
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Delete Report</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the report "{reportToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>
            Cancel
          </Button>
          <Button onClick={handleDeleteConfirm} color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notification */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={6000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        {notification ? (
          <Alert onClose={handleNotificationClose} severity={notification.type} sx={{ width: '100%' }}>
            {notification.message}
          </Alert>
        ) : null}
      </Snackbar>
    </ReportsContainer>
  );
};

export default ReportsListPageNew;
