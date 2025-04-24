import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, CircularProgress, Button, Alert, Divider } from '@mui/material';
import { Refresh as RefreshIcon, Settings as SettingsIcon } from '@mui/icons-material';
import DataObjectsCanvas from '../components/data-object-canvas/data-objects-canvas';
import { Link } from 'react-router-dom';
import useDataSourceStore from '../store/dataSourceStore';

const DataCanvasPage: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const Component = DataObjectsCanvas;
  
  // Get data sources from the store
  const { dataSources, fetchDataSources, loading: dataSourcesLoading } = useDataSourceStore();
  
  // Fetch data sources on component mount
  useEffect(() => {
    fetchDataSources();
  }, [fetchDataSources]);
  
  // Handle refresh button click
  const handleRefresh = () => {
    fetchDataSources();
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Data Object Canvas
          </Typography>
          <Typography variant="body1">
            Visualize and design data relationships between your database tables with this interactive canvas.
          </Typography>
        </Box>
        <Box>
          <Button 
            startIcon={<RefreshIcon />} 
            onClick={handleRefresh}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          <Button 
            component={Link} 
            to="/data-sources" 
            startIcon={<SettingsIcon />}
            variant="outlined"
          >
            Manage Data Sources
          </Button>
        </Box>
      </Box>
      
      {dataSources.length === 0 && !dataSourcesLoading && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body1" gutterBottom>
            No data sources found. You need to create a data source before you can use the Data Canvas.
          </Typography>
          <Button 
            component={Link} 
            to="/data-sources" 
            variant="contained" 
            size="small" 
            sx={{ mt: 1 }}
          >
            Create Data Source
          </Button>
        </Alert>
      )}
      
      <Divider sx={{ my: 2 }} />
      
      <Paper elevation={0} sx={{ height: 'calc(100vh - 250px)', overflow: 'hidden', position: 'relative' }}>
        {loading || dataSourcesLoading ? (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%' 
          }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Loading Data Canvas...</Typography>
          </Box>
        ) : error ? (
          <Box sx={{ p: 3 }}>
            <Typography color="error" variant="h6">Error</Typography>
            <Typography color="error">{error}</Typography>
            <Typography variant="body2" mt={2}>
              Please check your connection and try again.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              startIcon={<RefreshIcon />}
              onClick={handleRefresh}
              sx={{ mt: 2 }}
            >
              Retry
            </Button>
          </Box>
        ) : dataSources.length === 0 ? (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            textAlign: 'center'
          }}>
            <Typography variant="h6" gutterBottom>
              No Data Sources Available
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 500, mb: 3 }}>
              To use the Data Canvas, you need to create at least one data source.
              Go to the Data Sources page to create a new connection to your database.
            </Typography>
            <Button 
              component={Link} 
              to="/data-sources" 
              variant="contained" 
              color="primary"
              startIcon={<SettingsIcon />}
            >
              Manage Data Sources
            </Button>
          </Box>
        ) : Component ? (
          <Component />
        ) : null}
      </Paper>
    </Box>
  );
};

export default DataCanvasPage;
