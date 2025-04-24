import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Tooltip,
  Divider,
  Paper,
  Stack,
  Snackbar,
  Alert,
  SelectChangeEvent
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  MoreVert as MoreIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ContentCopy as DuplicateIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
  Share as ShareIcon,
  Public as PublicIcon,
  Lock as LockIcon
} from '@mui/icons-material';
import useReportStore from '../store/reportStore';
import { Report } from '../types/reporting';

const ReportsListPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    reports,
    fetchReports,
    deleteReport,
    resetReportBuilder,
    fetchDataSources,
    dataSources,
    loadingReports
  } = useReportStore();

  // Filter states
  const [searchText, setSearchText] = useState('');
  const [filterByDataSource, setFilterByDataSource] = useState<string>('');
  const [filterByType, setFilterByType] = useState<string>('all');

  // Menu states
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [activeReportId, setActiveReportId] = useState<string | null>(null);

  // Dialog states
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [reportToDelete, setReportToDelete] = useState<string | null>(null);

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

  // Load reports and data sources on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        await fetchDataSources();
        await fetchReports();
      } catch (error) {
        console.error('Error loading data:', error);
        setNotification({
          open: true,
          message: 'Failed to load reports. Please try again.',
          severity: 'error'
        });
      }
    };

    loadData();
  }, [fetchReports, fetchDataSources]);

  // Function to handle opening the report actions menu
  const handleMenuOpen = (event: React.MouseEvent<HTMLButtonElement>, reportId: string) => {
    setMenuAnchorEl(event.currentTarget);
    setActiveReportId(reportId);
  };

  // Function to handle closing the report actions menu
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setActiveReportId(null);
  };

  // Function to handle creating a new report
  const handleCreateNewReport = () => {
    resetReportBuilder();
    navigate('/reporting/reports/builder');
  };

  // Function to handle opening a report
  const handleOpenReport = (reportId: string) => {
    navigate(`/reporting/reports/${reportId}`);
  };

  // Function to handle editing a report
  const handleEditReport = (reportId: string) => {
    navigate(`/reporting/reports/builder?id=${reportId}`);
    handleMenuClose();
  };

  // Function to handle deleting a report
  const handleDeleteReport = () => {
    setReportToDelete(activeReportId);
    setDeleteDialogOpen(true);
    handleMenuClose();
  };

  // Function to confirm deleting a report
  const handleConfirmDelete = async () => {
    if (reportToDelete) {
      try {
        await deleteReport(reportToDelete);
        setNotification({
          open: true,
          message: 'Report deleted successfully',
          severity: 'success'
        });
      } catch (error) {
        console.error('Error deleting report:', error);
        setNotification({
          open: true,
          message: 'Failed to delete report',
          severity: 'error'
        });
      } finally {
        setDeleteDialogOpen(false);
        setReportToDelete(null);
      }
    }
  };

  // Function to handle duplicating a report
  const handleDuplicateReport = () => {
    // Implement report duplication functionality
    handleMenuClose();
    setNotification({
      open: true,
      message: 'Duplicate feature coming soon',
      severity: 'info'
    });
  };

  // Function to handle toggling a report as favorite
  const handleToggleFavorite = (reportId: string) => {
    // Implement favorite toggle functionality
    setNotification({
      open: true,
      message: 'Favorite toggling feature coming soon',
      severity: 'info'
    });
  };

  // Function to handle sharing a report
  const handleShareReport = () => {
    // Implement report sharing functionality
    handleMenuClose();
    setNotification({
      open: true,
      message: 'Sharing feature coming soon',
      severity: 'info'
    });
  };

  // Function to handle toggling a report's public status
  const handleTogglePublic = () => {
    // Implement public toggle functionality
    handleMenuClose();
    setNotification({
      open: true,
      message: 'Public toggle feature coming soon',
      severity: 'info'
    });
  };

  // Function to handle search and filtering of reports
  const getFilteredReports = (): Report[] => {
    return reports.filter(report => {
      // Text search
      const matchesSearch = searchText === '' ||
        report.name.toLowerCase().includes(searchText.toLowerCase()) ||
        (report.description && report.description.toLowerCase().includes(searchText.toLowerCase()));

      // Data source filter
      const matchesDataSource = filterByDataSource === '' || report.dataSourceId === filterByDataSource;

      // Type filter
      const matchesType = filterByType === 'all' ||
        (filterByType === 'public' && report.isPublic) ||
        (filterByType === 'private' && !report.isPublic) ||
        (filterByType === 'favorite' && report.isFavorite);

      return matchesSearch && matchesDataSource && matchesType;
    });
  };

  // Helper function to format dates
  const formatDate = (dateString: string | null | undefined) => {
    if (!dateString) {
      return 'N/A';
    }

    // Try to create a valid date object
    const date = new Date(dateString);

    // Check if the date is valid
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }

    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(date);
  };

  // Get the filtered reports
  const filteredReports = getFilteredReports();

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Reports</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleCreateNewReport}
        >
          Create New Report
        </Button>
      </Box>

      {/* Filters */}
      <Paper elevation={0} sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search Reports"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Data Source</InputLabel>
              <Select
                value={filterByDataSource}
                onChange={(e) => setFilterByDataSource(e.target.value)}
                label="Data Source"
              >
                <MenuItem value="">All Data Sources</MenuItem>
                {dataSources.map(dataSource => (
                  <MenuItem key={dataSource.id} value={dataSource.id}>
                    {dataSource.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Type</InputLabel>
              <Select
                value={filterByType}
                onChange={(e) => setFilterByType(e.target.value)}
                label="Type"
              >
                <MenuItem value="all">All Reports</MenuItem>
                <MenuItem value="public">Public Reports</MenuItem>
                <MenuItem value="private">Private Reports</MenuItem>
                <MenuItem value="favorite">Favorite Reports</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Chip
                label={`${filteredReports.length} Reports`}
                variant="outlined"
                sx={{ borderRadius: 1 }}
              />
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Reports Grid */}
      {loadingReports ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : filteredReports.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>No reports found</Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            Try adjusting your search filters or create a new report.
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={handleCreateNewReport}
            sx={{ mt: 2 }}
          >
            Create New Report
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {filteredReports.map(report => (
            <Grid item xs={12} md={4} key={report.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                    cursor: 'pointer'
                  }
                }}
              >
                <CardContent
                  onClick={() => handleOpenReport(report.id)}
                  sx={{ flexGrow: 1, pb: 1 }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
                      {report.name}
                    </Typography>
                    <Box>
                      {report.isPublic ? (
                        <Tooltip title="Public Report">
                          <PublicIcon fontSize="small" color="action" />
                        </Tooltip>
                      ) : (
                        <Tooltip title="Private Report">
                          <LockIcon fontSize="small" color="action" />
                        </Tooltip>
                      )}
                    </Box>
                  </Box>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      mt: 1,
                      mb: 2,
                      display: '-webkit-box',
                      overflow: 'hidden',
                      WebkitBoxOrient: 'vertical',
                      WebkitLineClamp: 2,
                      minHeight: '40px'
                    }}
                  >
                    {report.description || 'No description'}
                  </Typography>

                  <Stack direction="row" spacing={1}>
                    {dataSources.find(ds => ds.id === report.dataSourceId) && (
                      <Chip
                        label={dataSources.find(ds => ds.id === report.dataSourceId)?.name}
                        size="small"
                        variant="outlined"
                        sx={{ maxWidth: '150px' }}
                      />
                    )}
                  </Stack>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Last updated: {formatDate(report.updatedAt)}
                    </Typography>
                    {report.lastRunAt && (
                      <Typography variant="caption" color="text.secondary">
                        Last run: {formatDate(report.lastRunAt)}
                      </Typography>
                    )}
                  </Box>
                </CardContent>

                <Divider />

                <CardActions disableSpacing>
                  <IconButton
                    aria-label={report.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                    onClick={() => handleToggleFavorite(report.id)}
                    color={report.isFavorite ? 'secondary' : 'default'}
                  >
                    {report.isFavorite ? <FavoriteIcon /> : <FavoriteBorderIcon />}
                  </IconButton>
                  <IconButton aria-label="Edit report" onClick={() => handleEditReport(report.id)}>
                    <EditIcon />
                  </IconButton>
                  <Box sx={{ flexGrow: 1 }} />
                  <IconButton
                    aria-label="Report actions"
                    onClick={(e) => handleMenuOpen(e, report.id)}
                  >
                    <MoreIcon />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Actions Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleEditReport(activeReportId!)}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          Edit Report
        </MenuItem>
        <MenuItem onClick={handleDuplicateReport}>
          <ListItemIcon>
            <DuplicateIcon fontSize="small" />
          </ListItemIcon>
          Duplicate
        </MenuItem>
        <MenuItem onClick={handleShareReport}>
          <ListItemIcon>
            <ShareIcon fontSize="small" />
          </ListItemIcon>
          Share
        </MenuItem>
        <MenuItem onClick={handleTogglePublic}>
          <ListItemIcon>
            <PublicIcon fontSize="small" />
          </ListItemIcon>
          Make Public/Private
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleDeleteReport} sx={{ color: 'error.main' }}>
          <ListItemIcon sx={{ color: 'error.main' }}>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          Delete Report
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Report</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this report? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleConfirmDelete} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>

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

export default ReportsListPage;
