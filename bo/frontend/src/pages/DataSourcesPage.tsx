import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  FormHelperText,
  Snackbar,
  Alert,
  Chip,
  Tooltip,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
  Grid,
  Divider,
  InputAdornment
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Link as LinkIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  CloudUpload as CloudUploadIcon
} from '@mui/icons-material';
import useDataSourceStore from '../store/dataSourceStore';
import { DataSource, DataSourceType } from '../types/reporting';

// Component for data source management
const DataSourcesPage: React.FC = () => {
  const {
    dataSources,
    fetchDataSources,
    createDataSource,
    updateDataSource,
    deleteDataSource,
    testConnection,
    connectionTestStatus,
    resetConnectionTest,
    loading
  } = useDataSourceStore();

  // Dialog states
  const [openDialog, setOpenDialog] = useState(false);
  const [editingDataSource, setEditingDataSource] = useState<DataSource | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [dataSourceToDelete, setDataSourceToDelete] = useState<string | null>(null);

  // Form state
  const [formValues, setFormValues] = useState<Partial<DataSource>>({
    name: '',
    type: 'mysql',
    description: '',
    connection_string: '',
    database: '',
    username: '',
    password: '',
    host: '',
    port: '',
    ssl_enabled: false,
    options: {}
  });

  // File upload related states
  const [fileUploading, setFileUploading] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Form validation state
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

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

  // Load data sources on component mount
  useEffect(() => {
    fetchDataSources();
  }, [fetchDataSources]);

  // Reset form values when dialog is opened/closed or editing dataSource changes
  useEffect(() => {
    if (editingDataSource) {
      setFormValues({
        ...editingDataSource,
        // Don't include password in form values for security
        password: ''
      });
    } else {
      setFormValues({
        name: '',
        type: 'mysql',
        description: '',
        connection_string: '',
        database: '',
        username: '',
        password: '',
        host: '',
        port: '',
        ssl_enabled: false,
        options: {}
      });
    }

    // Reset form errors and connection test status
    setFormErrors({});
    resetConnectionTest();
  }, [editingDataSource, openDialog, resetConnectionTest]);

  // Handle opening the form dialog for creating a new data source
  const handleOpenCreateDialog = () => {
    setEditingDataSource(null);
    setOpenDialog(true);
  };

  // Handle opening the form dialog for editing an existing data source
  const handleOpenEditDialog = (dataSource: DataSource) => {
    setEditingDataSource(dataSource);
    setOpenDialog(true);
  };

  // Handle closing the form dialog
  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingDataSource(null);
    resetConnectionTest();
  };

  // Handle opening the delete confirmation dialog
  const handleOpenDeleteDialog = (dataSourceId: string) => {
    setDataSourceToDelete(dataSourceId);
    setDeleteConfirmOpen(true);
  };

  // Handle closing the delete confirmation dialog
  const handleCloseDeleteDialog = () => {
    setDeleteConfirmOpen(false);
    setDataSourceToDelete(null);
  };

  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | { name?: string; value: unknown }>) => {
    const { name, value } = e.target;
    if (!name) return;

    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear error for this field if it exists
    if (formErrors[name]) {
      setFormErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  // Handle file input change
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);

      // Update form values with file name
      setFormValues(prev => ({
        ...prev,
        database: file.name
      }));

      // Clear any existing errors for the database field
      if (formErrors.database) {
        setFormErrors(prev => ({
          ...prev,
          database: ''
        }));
      }
    }
  };

  // Handle file upload button click
  const handleFileUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle file upload
  const handleFileUpload = async () => {
    if (!selectedFile) return;

    setFileUploading(true);

    try {
      // Create FormData object to send file
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Here you would typically make an API call to upload the file
      // For example:
      // const response = await fetch('/api/upload', {
      //   method: 'POST',
      //   body: formData
      // });

      // For now, let's simulate a successful upload with a timeout
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Set success notification
      setNotification({
        open: true,
        message: 'File uploaded successfully',
        severity: 'success'
      });

      // Update form values with the file path from the server response
      // In a real implementation, you'd get the path from the API response
      setFormValues(prev => ({
        ...prev,
        database: `/uploads/${selectedFile.name}`
      }));

    } catch (error) {
      console.error('Error uploading file:', error);
      setNotification({
        open: true,
        message: `Failed to upload file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    } finally {
      setFileUploading(false);
    }
  };

  // Effect to trigger file upload when a file is selected
  useEffect(() => {
    if (selectedFile) {
      handleFileUpload();
    }
  }, [selectedFile]);

  // Validate form inputs
  const validateForm = (): boolean => {
    const errors: Record<string, string> = {};

    if (!formValues.name?.trim()) {
      errors.name = 'Name is required';
    }

    if (!formValues.type) {
      errors.type = 'Type is required';
    }

    // Different validation based on data source type
    if (formValues.type === 'mysql' || formValues.type === 'postgresql') {
      if (!formValues.host?.trim()) {
        errors.host = 'Host is required';
      }

      if (!formValues.username?.trim()) {
        errors.username = 'Username is required';
      }

      if (!editingDataSource && !formValues.password?.trim()) {
        errors.password = 'Password is required';
      }

      if (!formValues.database?.trim()) {
        errors.database = 'Database name is required';
      }

      if (formValues.port && !/^\d+$/.test(formValues.port)) {
        errors.port = 'Port must be a number';
      }
    } else if (formValues.type === 'sqlite') {
      if (!formValues.database?.trim()) {
        errors.database = 'Database path is required';
      }
    } else if (formValues.type === 'connection_string') {
      if (!formValues.connection_string?.trim()) {
        errors.connection_string = 'Connection string is required';
      }
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      // Check if user is logged in
      const authToken = localStorage.getItem('auth_token');
      const user = localStorage.getItem('user');

      console.log('Auth token exists:', !!authToken);
      console.log('User exists:', !!user);

      if (!authToken || !user) {
        setNotification({
          open: true,
          message: 'You must be logged in to create a data source',
          severity: 'error'
        });
        return;
      }

      // Transform form data to match the snake_case format expected by the API
      const transformedData: Partial<DataSource> = {
        name: formValues.name,
        description: formValues.description,
        connection_type: formValues.type, // Use snake_case field names for the API
        connection_config: {} as Record<string, any>,
      };

      // Add appropriate fields to connection_config based on the data source type
      switch (formValues.type) {
        case 'mysql':
        case 'postgresql':
          transformedData.connection_config = {
            host: formValues.host,
            port: formValues.port,
            database: formValues.database,
            username: formValues.username,
            password: formValues.password,
            ssl_enabled: formValues.ssl_enabled,
          };
          break;
        case 'sqlite':
          transformedData.connection_config = {
            database_path: formValues.database,
          };
          break;
        case 'csv':
        case 'excel':
          transformedData.connection_config = {
            file_path: formValues.database,
            has_header: true, // Default assumption
          };
          break;
        case 'connection_string':
          transformedData.connection_config = {
            connection_string: formValues.connection_string,
          };
          break;
        default:
          transformedData.connection_config = formValues.options || {};
      }

      console.log('Sending data source to backend (before type adapter):', JSON.stringify(transformedData, null, 2));

      if (editingDataSource) {
        // If editing, update existing data source
        await updateDataSource({
          ...transformedData,
          id: editingDataSource.id
        } as DataSource);

        setNotification({
          open: true,
          message: 'Data source updated successfully',
          severity: 'success'
        });
      } else {
        // If creating, create new data source
        await createDataSource(transformedData as DataSource);

        setNotification({
          open: true,
          message: 'Data source created successfully',
          severity: 'success'
        });
      }

      handleCloseDialog();
    } catch (error) {
      console.error('Error saving data source:', error);
      // Log more details about the error response
      if (error.response) {
        console.error('Error response data:', error.response.data);
        console.error('Error response status:', error.response.status);
        console.error('Error response headers:', error.response.headers);
      }

      setNotification({
        open: true,
        message: `Failed to save data source: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  };

  // Handle data source deletion
  const handleConfirmDelete = async () => {
    if (!dataSourceToDelete) return;

    try {
      await deleteDataSource(dataSourceToDelete);
      setNotification({
        open: true,
        message: 'Data source deleted successfully',
        severity: 'success'
      });
      handleCloseDeleteDialog();
    } catch (error) {
      console.error('Error deleting data source:', error);
      setNotification({
        open: true,
        message: `Failed to delete data source: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  };

  // Handle testing data source connection
  const handleTestConnection = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      await testConnection(formValues as DataSource);
    } catch (error) {
      console.error('Error testing connection:', error);
    }
  };

  // Get form fields based on data source type
  const renderFormFieldsByType = () => {
    const type = formValues.type;

    switch (type) {
      case 'mysql':
      case 'postgresql':
        return (
          <>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Host"
                name="host"
                value={formValues.host || ''}
                onChange={handleInputChange}
                error={!!formErrors.host}
                helperText={formErrors.host || ''}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Port"
                name="port"
                value={formValues.port || ''}
                onChange={handleInputChange}
                error={!!formErrors.port}
                helperText={formErrors.port || `Default: ${type === 'mysql' ? '3306' : '5432'}`}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Database"
                name="database"
                value={formValues.database || ''}
                onChange={handleInputChange}
                error={!!formErrors.database}
                helperText={formErrors.database || ''}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Username"
                name="username"
                value={formValues.username || ''}
                onChange={handleInputChange}
                error={!!formErrors.username}
                helperText={formErrors.username || ''}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="password"
                label="Password"
                name="password"
                value={formValues.password || ''}
                onChange={handleInputChange}
                error={!!formErrors.password}
                helperText={editingDataSource ? 'Leave blank to keep current password' : formErrors.password || ''}
                required={!editingDataSource}
              />
            </Grid>
          </>
        );

      case 'sqlite':
        return (
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Database Path"
              name="database"
              value={formValues.database || ''}
              onChange={handleInputChange}
              error={!!formErrors.database}
              helperText={formErrors.database || 'Path to SQLite database file'}
              required
            />
          </Grid>
        );

      case 'csv':
      case 'excel':
        return (
          <Grid item xs={12}>
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept={type === 'csv' ? '.csv' : '.xlsx,.xls'}
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />

            <TextField
              fullWidth
              label="File Path"
              name="database"
              value={formValues.database || ''}
              onChange={handleInputChange}
              error={!!formErrors.database}
              helperText={formErrors.database || `Path to ${type.toUpperCase()} file`}
              required
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <Button
                      color="primary"
                      onClick={handleFileUploadClick}
                      disabled={fileUploading}
                      startIcon={fileUploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
                    >
                      {fileUploading ? 'Uploading...' : 'Upload'}
                    </Button>
                  </InputAdornment>
                ),
              }}
            />
            {selectedFile && (
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Selected file: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
              </Typography>
            )}
          </Grid>
        );

      case 'connection_string':
        return (
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Connection String"
              name="connection_string"
              value={formValues.connection_string || ''}
              onChange={handleInputChange}
              error={!!formErrors.connection_string}
              helperText={formErrors.connection_string || 'e.g., mysql://user:password@localhost:3306/database'}
              required
            />
          </Grid>
        );

      default:
        return null;
    }
  };

  const getDataSourceTypeLabel = (type: string): string => {
    const typeMap: Record<string, string> = {
      mysql: 'MySQL',
      postgresql: 'PostgreSQL',
      sqlite: 'SQLite',
      csv: 'CSV File',
      excel: 'Excel File',
      bigquery: 'BigQuery',
      snowflake: 'Snowflake',
      redshift: 'Amazon Redshift',
      connection_string: 'Connection String',
      api: 'API',
      custom: 'Custom'
    };

    return typeMap[type] || type;
  };

  const getDataSourceStatusColor = (status: string): string => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getDataSourceStatusLabel = (status: string): string => {
    switch (status) {
      case 'active':
        return 'Active';
      case 'inactive':
        return 'Inactive';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  // Render the component
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Data Sources</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={handleOpenCreateDialog}
        >
          Add Data Source
        </Button>
      </Box>

      {/* Data Sources Grid */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : dataSources.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>No data sources found</Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            Add a data source to start creating reports.
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={handleOpenCreateDialog}
            sx={{ mt: 2 }}
          >
            Add Data Source
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {dataSources.map((dataSource) => (
            <Grid item xs={12} md={6} lg={4} key={dataSource.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      {dataSource.name}
                    </Typography>
                    <Chip
                      size="small"
                      label={getDataSourceStatusLabel(dataSource.status || 'active')}
                      color={getDataSourceStatusColor(dataSource.status || 'active') as any}
                    />
                  </Box>

                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {getDataSourceTypeLabel(dataSource.type)}
                  </Typography>

                  {dataSource.description && (
                    <Typography
                      variant="body2"
                      sx={{
                        mt: 1,
                        mb: 2,
                        display: '-webkit-box',
                        overflow: 'hidden',
                        WebkitBoxOrient: 'vertical',
                        WebkitLineClamp: 2,
                      }}
                    >
                      {dataSource.description}
                    </Typography>
                  )}

                  <Box sx={{ mt: 1 }}>
                    {dataSource.type === 'mysql' || dataSource.type === 'postgresql' ? (
                      <Typography variant="caption" display="block">
                        {dataSource.host}{dataSource.port ? `:${dataSource.port}` : ''} / {dataSource.database}
                      </Typography>
                    ) : dataSource.type === 'sqlite' || dataSource.type === 'csv' || dataSource.type === 'excel' ? (
                      <Typography variant="caption" display="block" sx={{ wordBreak: 'break-all' }}>
                        {dataSource.database}
                      </Typography>
                    ) : null}
                  </Box>
                </CardContent>

                <Divider />

                <CardActions>
                  <Tooltip title="Edit">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenEditDialog(dataSource)}
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Delete">
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDeleteDialog(dataSource.id)}
                      color="error"
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>

                  <Box sx={{ flexGrow: 1 }} />

                  <Tooltip title="Last updated">
                    <Typography variant="caption" color="text.secondary">
                      {new Date(dataSource.updated_at).toLocaleDateString()}
                    </Typography>
                  </Tooltip>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Data Source Form Dialog */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: { overflowY: 'visible' }
        }}
      >
        <DialogTitle>
          {editingDataSource ? 'Edit Data Source' : 'Add Data Source'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Name"
                name="name"
                value={formValues.name || ''}
                onChange={handleInputChange}
                error={!!formErrors.name}
                helperText={formErrors.name || ''}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth error={!!formErrors.type} required>
                <InputLabel>Type</InputLabel>
                <Select
                  name="type"
                  value={formValues.type || ''}
                  onChange={handleInputChange as any}
                  label="Type"
                >
                  <MenuItem value="mysql">MySQL</MenuItem>
                  <MenuItem value="postgresql">PostgreSQL</MenuItem>
                  <MenuItem value="sqlite">SQLite</MenuItem>
                  <MenuItem value="csv">CSV File</MenuItem>
                  <MenuItem value="excel">Excel File</MenuItem>
                  <MenuItem value="bigquery">BigQuery</MenuItem>
                  <MenuItem value="connection_string">Connection String</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
                {formErrors.type && <FormHelperText>{formErrors.type}</FormHelperText>}
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                name="description"
                value={formValues.description || ''}
                onChange={handleInputChange}
                multiline
                rows={2}
              />
            </Grid>

            {/* Render fields based on data source type */}
            {renderFormFieldsByType()}

            {/* Connection test result */}
            {connectionTestStatus.tested && (
              <Grid item xs={12}>
                <Paper
                  sx={{
                    p: 2,
                    display: 'flex',
                    alignItems: 'center',
                    bgcolor: connectionTestStatus.success ? 'success.light' : 'error.light',
                    color: 'common.white'
                  }}
                >
                  {connectionTestStatus.success ? (
                    <CheckCircleIcon sx={{ mr: 1 }} />
                  ) : (
                    <ErrorIcon sx={{ mr: 1 }} />
                  )}
                  <Box>
                    <Typography variant="body1">
                      {connectionTestStatus.success ? 'Connection successful!' : 'Connection failed'}
                    </Typography>
                    {connectionTestStatus.message && (
                      <Typography variant="body2">
                        {connectionTestStatus.message}
                      </Typography>
                    )}
                  </Box>
                </Paper>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button
            onClick={handleTestConnection}
            startIcon={<LinkIcon />}
            color="primary"
          >
            Test Connection
          </Button>
          <Button
            onClick={handleSubmit}
            variant="contained"
            color="primary"
            disabled={connectionTestStatus.tested && !connectionTestStatus.success}
          >
            {editingDataSource ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmOpen}
        onClose={handleCloseDeleteDialog}
      >
        <DialogTitle>Delete Data Source</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this data source? This action cannot be undone.
            Any reports using this data source will no longer work.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
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

export default DataSourcesPage;