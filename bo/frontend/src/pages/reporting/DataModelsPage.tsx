import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Menu,
  MenuItem,
  ListItemIcon,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  CircularProgress,
  Alert,
  Tooltip
} from '@mui/material';
import {
  Search as SearchIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  MoreVert as MoreVertIcon,
  Visibility as ViewIcon,
  ContentCopy as DuplicateIcon,
  Code as CodeIcon,
  Storage as StorageIcon,
  Link as LinkIcon
} from '@mui/icons-material';
import useDataModelsStore from '../../store/dataModelsStore';
import useDataSourceStore from '../../store/dataSourceStore';
import { DataModel } from '../../components/data-object-canvas/data-objects-api';
import { formatDistanceToNow } from 'date-fns';

const DataModelsPage: React.FC = () => {
  const navigate = useNavigate();
  const { 
    dataModels, 
    fetchDataModels, 
    deleteDataModel, 
    isLoading, 
    error 
  } = useDataModelsStore();
  
  const { 
    dataSources, 
    fetchDataSources 
  } = useDataSourceStore();
  
  // Local state
  const [searchText, setSearchText] = useState('');
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<DataModel | null>(null);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // Load data models and data sources on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        await fetchDataSources();
        await fetchDataModels();
      } catch (error) {
        console.error('Error loading data:', error);
        setNotification({
          open: true,
          message: 'Failed to load data models. Please try again.',
          severity: 'error'
        });
      }
    };
    
    loadData();
  }, [fetchDataModels, fetchDataSources]);
  
  // Filter data models based on search text
  const filteredModels = dataModels.filter(model => 
    model.name.toLowerCase().includes(searchText.toLowerCase())
  );
  
  // Function to handle opening the model actions menu
  const handleMenuOpen = (event: React.MouseEvent<HTMLButtonElement>, modelId: string) => {
    setMenuAnchorEl(event.currentTarget);
    setActiveModelId(modelId);
  };
  
  // Function to handle closing the model actions menu
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setActiveModelId(null);
  };
  
  // Function to handle creating a new data model
  const handleCreateNewModel = () => {
    navigate('/reporting/data-canvas');
  };
  
  // Function to handle viewing a data model
  const handleViewModel = (modelId: string) => {
    navigate(`/reporting/data-canvas?modelId=${modelId}`);
    handleMenuClose();
  };
  
  // Function to handle editing a data model
  const handleEditModel = (modelId: string) => {
    navigate(`/reporting/data-canvas?modelId=${modelId}&edit=true`);
    handleMenuClose();
  };
  
  // Function to handle duplicating a data model
  const handleDuplicateModel = () => {
    // TODO: Implement duplicating a data model
    handleMenuClose();
  };
  
  // Function to handle generating SQL from a data model
  const handleGenerateSQL = () => {
    // TODO: Implement generating SQL from a data model
    handleMenuClose();
  };
  
  // Function to handle opening the delete confirmation dialog
  const handleDeleteClick = () => {
    const model = dataModels.find(m => m.id === activeModelId);
    if (model) {
      setModelToDelete(model);
      setDeleteDialogOpen(true);
    }
    handleMenuClose();
  };
  
  // Function to handle canceling the delete operation
  const handleCancelDelete = () => {
    setDeleteDialogOpen(false);
    setModelToDelete(null);
  };
  
  // Function to handle confirming the delete operation
  const handleConfirmDelete = async () => {
    if (modelToDelete) {
      try {
        await deleteDataModel(modelToDelete.id);
        setNotification({
          open: true,
          message: `Data model "${modelToDelete.name}" deleted successfully.`,
          severity: 'success'
        });
      } catch (error) {
        console.error('Error deleting data model:', error);
        setNotification({
          open: true,
          message: 'Failed to delete data model. Please try again.',
          severity: 'error'
        });
      }
    }
    setDeleteDialogOpen(false);
    setModelToDelete(null);
  };
  
  // Function to get data source name by ID
  const getDataSourceName = (dataSourceId: string) => {
    const dataSource = dataSources.find(ds => ds.id === dataSourceId);
    return dataSource ? dataSource.name : 'Unknown Data Source';
  };
  
  // Function to format date
  const formatDate = (dateString: string) => {
    try {
      return formatDistanceToNow(new Date(dateString), { addSuffix: true });
    } catch (error) {
      return 'Unknown date';
    }
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Data Models</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleCreateNewModel}
        >
          Create New Model
        </Button>
      </Box>
      
      {/* Filters */}
      <Paper elevation={0} sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search Data Models"
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
        </Grid>
      </Paper>
      
      {/* Data Models Grid */}
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      ) : filteredModels.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>No data models found</Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            Try adjusting your search filters or create a new data model.
          </Typography>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={handleCreateNewModel}
            sx={{ mt: 2 }}
          >
            Create New Data Model
          </Button>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {filteredModels.map(model => (
            <Grid item xs={12} md={4} key={model.id}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  '&:hover': {
                    boxShadow: 3
                  }
                }}
              >
                <CardContent 
                  onClick={() => handleViewModel(model.id)}
                  sx={{ flexGrow: 1, pb: 1, cursor: 'pointer' }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
                      {model.name}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleMenuOpen(e, model.id);
                      }}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, mb: 2 }}>
                    <StorageIcon fontSize="small" color="action" sx={{ mr: 0.5 }} />
                    <Typography variant="body2" color="text.secondary">
                      {getDataSourceName(model.dataSourceId)}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                    <Chip 
                      size="small" 
                      label={`${model.tables.length} Tables`} 
                      variant="outlined" 
                    />
                    <Chip 
                      size="small" 
                      label={`${model.relationships.length} Relationships`} 
                      variant="outlined" 
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 'auto' }}>
                    <Typography variant="caption" color="text.secondary">
                      Created {formatDate(model.createdAt)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Updated {formatDate(model.updatedAt)}
                    </Typography>
                  </Box>
                </CardContent>
                <CardActions>
                  <Button 
                    size="small" 
                    startIcon={<ViewIcon />}
                    onClick={() => handleViewModel(model.id)}
                  >
                    View
                  </Button>
                  <Button 
                    size="small" 
                    startIcon={<EditIcon />}
                    onClick={() => handleEditModel(model.id)}
                  >
                    Edit
                  </Button>
                  <Button 
                    size="small" 
                    color="error" 
                    startIcon={<DeleteIcon />}
                    onClick={(e) => {
                      e.stopPropagation();
                      setActiveModelId(model.id);
                      handleDeleteClick();
                    }}
                  >
                    Delete
                  </Button>
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
        <MenuItem onClick={() => activeModelId && handleViewModel(activeModelId)}>
          <ListItemIcon>
            <ViewIcon fontSize="small" />
          </ListItemIcon>
          View Model
        </MenuItem>
        <MenuItem onClick={() => activeModelId && handleEditModel(activeModelId)}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          Edit Model
        </MenuItem>
        <MenuItem onClick={handleDuplicateModel}>
          <ListItemIcon>
            <DuplicateIcon fontSize="small" />
          </ListItemIcon>
          Duplicate
        </MenuItem>
        <MenuItem onClick={handleGenerateSQL}>
          <ListItemIcon>
            <CodeIcon fontSize="small" />
          </ListItemIcon>
          Generate SQL
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleDeleteClick} sx={{ color: 'error.main' }}>
          <ListItemIcon sx={{ color: 'error.main' }}>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          Delete
        </MenuItem>
      </Menu>
      
      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleCancelDelete}
      >
        <DialogTitle>Delete Data Model</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the data model "{modelToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelDelete}>Cancel</Button>
          <Button onClick={handleConfirmDelete} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Notification Snackbar */}
      {notification.open && (
        <Alert 
          severity={notification.severity}
          sx={{ 
            position: 'fixed', 
            bottom: 16, 
            right: 16, 
            zIndex: 9999,
            boxShadow: 3
          }}
          onClose={() => setNotification({ ...notification, open: false })}
        >
          {notification.message}
        </Alert>
      )}
    </Box>
  );
};

export default DataModelsPage;
