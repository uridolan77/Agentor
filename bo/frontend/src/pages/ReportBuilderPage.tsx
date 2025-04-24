import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Button,
  TextField,
  Tabs,
  Tab,
  FormControlLabel,
  Switch,
  Divider,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  IconButton,
  MenuItem,
  Tooltip,
  Alert
} from '@mui/material';
import {
  Save as SaveIcon,
  Preview as PreviewIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  ArrowBack as ArrowBackIcon,
  Public as PublicIcon,
  Lock as LockIcon
} from '@mui/icons-material';
import useReportStore from '../store/reportStore';
import { DataSource, Dimension, Metric, CalculatedMetric } from '../types/reporting';

// Define tab panel props
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

// Tab panel component
const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`report-builder-tabpanel-${index}`}
      aria-labelledby={`report-builder-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

// Visualization type options - must match the types in VisualizationConfig
const VISUALIZATION_TYPES = [
  { value: 'bar', label: 'Bar Chart' },
  { value: 'line', label: 'Line Chart' },
  { value: 'pie', label: 'Pie Chart' },
  { value: 'table', label: 'Table' },
  { value: 'area', label: 'Area Chart' },
  { value: 'scatter', label: 'Scatter Plot' },
  { value: 'metric', label: 'KPI Metric' }
];

const ReportBuilderPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const reportId = searchParams.get('id');
  const [error, setError] = useState<string | null>(null);
  
  const {
    selectedReport,
    reportName,
    reportDescription,
    reportBuilderFields,
    reportBuilderFilters,
    reportBuilderSorting,
    reportBuilderVisualizations,
    reportBuilderLayout,
    dataSources,
    dimensions,
    metrics,
    calculatedMetrics,
    selectedDataSourceId,
    fetchReportById,
    fetchDataSources,
    fetchFields,
    resetReportBuilder,
    createReport,
    updateReport,
    setReportName,
    setReportDescription,
    addVisualization,
    updateVisualization,
    removeVisualization,
    selectDataSource,
    loadingDataSources,
    loadingReports
  } = useReportStore();
  
  const [tabValue, setTabValue] = useState(0);
  const [discardDialogOpen, setDiscardDialogOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [loadingData, setLoadingData] = useState(true);
  
  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoadingData(true);
        setError(null);
        await fetchDataSources();
        
        if (reportId) {
          try {
            await fetchReportById(reportId);
          } catch (error) {
            console.error('Error fetching report:', error);
            setError('Failed to load the report. It may have been deleted or you do not have permission to access it.');
          }
        } else {
          resetReportBuilder();
        }
      } catch (error) {
        console.error('Error loading data:', error);
        setError('Failed to load data. Please try again later.');
      } finally {
        setLoadingData(false);
      }
    };
    
    loadData();
  }, [reportId, fetchDataSources, fetchReportById, resetReportBuilder]);
  
  // When data source changes, fetch fields
  useEffect(() => {
    if (selectedDataSourceId) {
      fetchFields(selectedDataSourceId).catch(error => {
        console.error('Error loading fields:', error);
      });
    }
  }, [selectedDataSourceId, fetchFields]);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    if (name === 'name') {
      setReportName(value);
    } else if (name === 'description') {
      setReportDescription(value);
    }
    
    // Clear errors
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };
  
  const handleSwitchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    // Handle switch changes based on name
    // This would be implemented to update the specific field
  };
  
  const handleDataSourceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const dsId = e.target.value;
    selectDataSource(dsId);
    
    // Clear errors
    if (errors.dataSourceId) {
      setErrors(prev => ({ ...prev, dataSourceId: '' }));
    }
  };
  
  const handleAddVisualization = () => {
    // Explicitly type the visualization to match VisualizationConfig
    const newViz = {
      id: `viz-${Date.now()}`,
      title: 'New Visualization',
      type: 'bar' as 'table' | 'bar' | 'line' | 'pie' | 'area' | 'scatter' | 'metric',
      dimensions: [],
      metrics: [],
      settings: {}
    };
    
    addVisualization(newViz);
  };
  
  const handleUpdateVisualization = (
    id: string,
    updates: Partial<any>
  ) => {
    updateVisualization(id, updates);
    
    // If dimensions or metrics were updated, also update the report fields
    if (updates.dimensions || updates.metrics) {
      const updatedViz = reportBuilderVisualizations.find(v => v.id === id);
      if (updatedViz) {
        // Create or update report fields based on dimensions and metrics
        const dimensionFields = (updates.dimensions || updatedViz.dimensions || []).map(dimId => ({
          id: dimId,
          type: 'dimension',
          name: dimensions.find(d => d.id === dimId)?.name || dimId
        }));
        
        const metricFields = (updates.metrics || updatedViz.metrics || []).map(metricId => ({
          id: metricId,
          type: 'metric',
          name: metrics.find(m => m.id === metricId)?.name || 
                calculatedMetrics.find(m => m.id === metricId)?.name || 
                metricId
        }));
        
        // This would need additional logic to avoid duplicate fields
        // For now, we'll just ensure the visualization's dimensions and metrics are in the fields
      }
    }
  };
  
  const handleRemoveVisualization = (id: string) => {
    removeVisualization(id);
  };
  
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!reportName?.trim()) {
      newErrors.name = 'Report name is required';
    }
    
    if (!selectedDataSourceId) {
      newErrors.dataSourceId = 'Data source is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }
    
    setSaving(true);
    
    try {
      if (reportId && selectedReport) {
        await updateReport();
      } else {
        await createReport();
      }
      setSaving(false);
      setSaveDialogOpen(true);
    } catch (error) {
      console.error('Error saving report:', error);
      setSaving(false);
      setError('Failed to save report. Please try again.');
    }
  };
  
  const handlePreview = async () => {
    if (!validateForm()) {
      return;
    }
    
    try {
      let reportToPreview;
      if (reportId && selectedReport) {
        reportToPreview = await updateReport();
      } else {
        reportToPreview = await createReport();
      }
      
      navigate(`/reports/${reportToPreview.id}`);
    } catch (error) {
      console.error('Error saving report for preview:', error);
      setError('Failed to create preview. Please try again.');
    }
  };
  
  const handleOpenDiscardDialog = () => {
    setDiscardDialogOpen(true);
  };
  
  const handleCloseDiscardDialog = () => {
    setDiscardDialogOpen(false);
  };
  
  const handleDiscard = () => {
    navigate('/reports');
  };
  
  const handleCloseSaveDialog = () => {
    setSaveDialogOpen(false);
    navigate('/reports');
  };
  
  // Show loading state
  if (loadingData || loadingDataSources || loadingReports) {
    return (
      <Container>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ ml: 2 }}>
            Loading report builder...
          </Typography>
        </Box>
      </Container>
    );
  }

  // Show error state
  if (error) {
    return (
      <Container>
        <Paper sx={{ p: 4, textAlign: 'center', mt: 4 }}>
          <Typography variant="h6" color="error" gutterBottom>
            Error
          </Typography>
          <Typography variant="body1" paragraph>
            {error}
          </Typography>
          <Button
            variant="contained"
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/reports')}
          >
            Back to Reports
          </Button>
        </Paper>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header with back button and actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton onClick={handleOpenDiscardDialog} sx={{ mr: 1 }}>
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            {reportId ? 'Edit Report' : 'New Report'}
          </Typography>
        </Box>
        <Box>
          <Button
            variant="outlined"
            startIcon={<PreviewIcon />}
            onClick={handlePreview}
            sx={{ mr: 1 }}
          >
            Preview
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? <CircularProgress size={24} /> : 'Save Report'}
          </Button>
        </Box>
      </Box>
      
      {/* Tabs for different sections */}
      <Paper sx={{ mb: 4 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="report builder tabs">
            <Tab label="General" id="report-builder-tab-0" />
            <Tab label="Visualizations" id="report-builder-tab-1" />
            <Tab label="Layout" id="report-builder-tab-2" />
            <Tab label="Sharing" id="report-builder-tab-3" />
          </Tabs>
        </Box>
        
        {/* General Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                required
                label="Report Name"
                name="name"
                value={reportName || ''}
                onChange={handleInputChange}
                error={!!errors.name}
                helperText={errors.name}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                required
                label="Data Source"
                name="dataSourceId"
                value={selectedDataSourceId || ''}
                onChange={handleDataSourceChange}
                error={!!errors.dataSourceId}
                helperText={errors.dataSourceId}
              >
                <MenuItem value="">Select a data source</MenuItem>
                {dataSources.map((ds) => (
                  <MenuItem key={ds.id} value={ds.id}>
                    {ds.name}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Description"
                name="description"
                value={reportDescription || ''}
                onChange={handleInputChange}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={selectedReport?.isPublic || false}
                    onChange={(e) => e} // This needs to be implemented to update the report public state
                    name="isPublic"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {selectedReport?.isPublic ? (
                      <PublicIcon fontSize="small" sx={{ mr: 1 }} />
                    ) : (
                      <LockIcon fontSize="small" sx={{ mr: 1 }} />
                    )}
                    {selectedReport?.isPublic ? 'Public' : 'Private'} Report
                  </Box>
                }
              />
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Visualizations Tab */}
        <TabPanel value={tabValue} index={1}>
          {!selectedDataSourceId ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              Please select a data source in the General tab first.
            </Alert>
          ) : (
            <>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3 }}>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={handleAddVisualization}
                >
                  Add Visualization
                </Button>
              </Box>
              
              {reportBuilderVisualizations.length === 0 ? (
                <Paper sx={{ p: 4, textAlign: 'center' }}>
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No visualizations added yet
                  </Typography>
                  <Typography variant="body1" color="text.secondary" paragraph>
                    Add a visualization to start building your report
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={handleAddVisualization}
                  >
                    Add Visualization
                  </Button>
                </Paper>
              ) : (
                <Grid container spacing={3}>
                  {reportBuilderVisualizations.map((viz, index) => (
                    <Grid item xs={12} key={viz.id}>
                      <Paper sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Typography variant="h6">
                            Visualization {index + 1}
                          </Typography>
                          <IconButton
                            color="error"
                            onClick={() => handleRemoveVisualization(viz.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Box>
                        
                        <Grid container spacing={3}>
                          <Grid item xs={12} md={6}>
                            <TextField
                              fullWidth
                              label="Title"
                              value={viz.title}
                              onChange={(e) => handleUpdateVisualization(viz.id, { title: e.target.value })}
                            />
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <TextField
                              select
                              fullWidth
                              label="Visualization Type"
                              value={viz.type}
                              onChange={(e) => handleUpdateVisualization(viz.id, { type: e.target.value })}
                            >
                              {VISUALIZATION_TYPES.map((type) => (
                                <MenuItem key={type.value} value={type.value}>
                                  {type.label}
                                </MenuItem>
                              ))}
                            </TextField>
                          </Grid>
                          
                          <Grid item xs={12}>
                            <Divider sx={{ my: 1 }} />
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Dimensions
                            </Typography>
                            {dimensions.length === 0 ? (
                              <Typography variant="body2" color="text.secondary">
                                No dimensions available for this data source
                              </Typography>
                            ) : (
                              <TextField
                                select
                                fullWidth
                                label="Select Dimensions"
                                SelectProps={{ multiple: true }}
                                value={viz.dimensions || []}
                                onChange={(e) => {
                                  // Handle the type conversion properly for multi-select
                                  const values = Array.isArray(e.target.value) ? e.target.value : [e.target.value];
                                  handleUpdateVisualization(viz.id, { dimensions: values });
                                }}
                              >
                                {dimensions.map((dim) => (
                                  <MenuItem key={dim.id} value={dim.id}>
                                    {dim.name}
                                  </MenuItem>
                                ))}
                              </TextField>
                            )}
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Metrics
                            </Typography>
                            <TextField
                              select
                              fullWidth
                              label="Select Metrics"
                              SelectProps={{ multiple: true }}
                              value={viz.metrics || []}
                              onChange={(e) => {
                                // Handle the type conversion properly for multi-select
                                const values = Array.isArray(e.target.value) ? e.target.value : [e.target.value];
                                handleUpdateVisualization(viz.id, { metrics: values });
                              }}
                            >
                              {metrics.map((metric) => (
                                <MenuItem key={metric.id} value={metric.id}>
                                  {metric.name}
                                </MenuItem>
                              ))}
                              {calculatedMetrics.map((metric) => (
                                <MenuItem key={metric.id} value={metric.id}>
                                  {metric.name} (Calculated)
                                </MenuItem>
                              ))}
                            </TextField>
                          </Grid>
                          
                          {/* Table-specific configuration options */}
                          {viz.type === 'table' && (
                            <>
                              <Grid item xs={12}>
                                <Divider sx={{ my: 2 }}>
                                  <Typography variant="subtitle2">Table Configuration</Typography>
                                </Divider>
                              </Grid>
                              
                              <Grid item xs={12} md={6}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Table Features
                                </Typography>
                                <FormControlLabel
                                  control={
                                    <Switch
                                      checked={viz.settings?.pagination ?? true}
                                      onChange={(e) => handleUpdateVisualization(viz.id, { 
                                        settings: { 
                                          ...viz.settings, 
                                          pagination: e.target.checked 
                                        } 
                                      })}
                                    />
                                  }
                                  label="Enable Pagination"
                                />
                                <FormControlLabel
                                  control={
                                    <Switch
                                      checked={viz.settings?.sorting ?? true}
                                      onChange={(e) => handleUpdateVisualization(viz.id, { 
                                        settings: { 
                                          ...viz.settings, 
                                          sorting: e.target.checked 
                                        } 
                                      })}
                                    />
                                  }
                                  label="Enable Sorting"
                                />
                                <FormControlLabel
                                  control={
                                    <Switch
                                      checked={viz.settings?.filtering ?? false}
                                      onChange={(e) => handleUpdateVisualization(viz.id, { 
                                        settings: { 
                                          ...viz.settings, 
                                          filtering: e.target.checked 
                                        } 
                                      })}
                                    />
                                  }
                                  label="Enable Filtering"
                                />
                              </Grid>
                              
                              <Grid item xs={12} md={6}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Display Options
                                </Typography>
                                <TextField
                                  fullWidth
                                  type="number"
                                  label="Default Rows Per Page"
                                  value={viz.settings?.rowsPerPage ?? 10}
                                  onChange={(e) => handleUpdateVisualization(viz.id, { 
                                    settings: { 
                                      ...viz.settings, 
                                      rowsPerPage: parseInt(e.target.value) || 10 
                                    } 
                                  })}
                                  InputProps={{ inputProps: { min: 5, max: 100 } }}
                                  sx={{ mb: 2 }}
                                />
                                <FormControlLabel
                                  control={
                                    <Switch
                                      checked={viz.settings?.striped ?? true}
                                      onChange={(e) => handleUpdateVisualization(viz.id, { 
                                        settings: { 
                                          ...viz.settings, 
                                          striped: e.target.checked 
                                        } 
                                      })}
                                    />
                                  }
                                  label="Striped Rows"
                                />
                                <FormControlLabel
                                  control={
                                    <Switch
                                      checked={viz.settings?.bordered ?? false}
                                      onChange={(e) => handleUpdateVisualization(viz.id, { 
                                        settings: { 
                                          ...viz.settings, 
                                          bordered: e.target.checked 
                                        } 
                                      })}
                                    />
                                  }
                                  label="Bordered Cells"
                                />
                              </Grid>
                              
                              <Grid item xs={12}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Column Configuration
                                </Typography>
                                <Alert severity="info" sx={{ mb: 2 }}>
                                  You can configure column settings for each dimension and metric added to this table.
                                </Alert>
                                
                                {/* Dimension columns configuration */}
                                {viz.dimensions && viz.dimensions.length > 0 && (
                                  <Paper sx={{ p: 2, mb: 2 }} variant="outlined">
                                    <Typography variant="subtitle2" gutterBottom>
                                      Dimension Columns
                                    </Typography>
                                    
                                    {viz.dimensions.map((dimId: string) => {
                                      const dimension = dimensions.find(d => d.id === dimId);
                                      const columnSetting = viz.settings?.columns?.[dimId] || {};
                                      
                                      return dimension ? (
                                        <Box key={dimId} sx={{ mb: 2 }}>
                                          <Typography variant="body2" fontWeight="bold">
                                            {dimension.name}
                                          </Typography>
                                          <Grid container spacing={2} alignItems="center">
                                            <Grid item xs={6} md={4}>
                                              <TextField
                                                fullWidth
                                                label="Display Name"
                                                size="small"
                                                value={columnSetting.displayName || dimension.name}
                                                onChange={(e) => {
                                                  const updatedColumns = {
                                                    ...viz.settings?.columns,
                                                    [dimId]: {
                                                      ...columnSetting,
                                                      displayName: e.target.value
                                                    }
                                                  };
                                                  handleUpdateVisualization(viz.id, { 
                                                    settings: { 
                                                      ...viz.settings, 
                                                      columns: updatedColumns
                                                    } 
                                                  });
                                                }}
                                              />
                                            </Grid>
                                            <Grid item xs={6} md={4}>
                                              <TextField
                                                select
                                                fullWidth
                                                label="Alignment"
                                                size="small"
                                                value={columnSetting.align || 'left'}
                                                onChange={(e) => {
                                                  const updatedColumns = {
                                                    ...viz.settings?.columns,
                                                    [dimId]: {
                                                      ...columnSetting,
                                                      align: e.target.value
                                                    }
                                                  };
                                                  handleUpdateVisualization(viz.id, { 
                                                    settings: { 
                                                      ...viz.settings, 
                                                      columns: updatedColumns
                                                    } 
                                                  });
                                                }}
                                              >
                                                <MenuItem value="left">Left</MenuItem>
                                                <MenuItem value="center">Center</MenuItem>
                                                <MenuItem value="right">Right</MenuItem>
                                              </TextField>
                                            </Grid>
                                            <Grid item xs={12} md={4}>
                                              <FormControlLabel
                                                control={
                                                  <Switch
                                                    checked={columnSetting.sortable !== false}
                                                    onChange={(e) => {
                                                      const updatedColumns = {
                                                        ...viz.settings?.columns,
                                                        [dimId]: {
                                                          ...columnSetting,
                                                          sortable: e.target.checked
                                                        }
                                                      };
                                                      handleUpdateVisualization(viz.id, { 
                                                        settings: { 
                                                          ...viz.settings, 
                                                          columns: updatedColumns
                                                        } 
                                                      });
                                                    }}
                                                  />
                                                }
                                                label="Sortable"
                                              />
                                            </Grid>
                                          </Grid>
                                        </Box>
                                      ) : null;
                                    })}
                                  </Paper>
                                )}
                                
                                {/* Metric columns configuration */}
                                {viz.metrics && viz.metrics.length > 0 && (
                                  <Paper sx={{ p: 2 }} variant="outlined">
                                    <Typography variant="subtitle2" gutterBottom>
                                      Metric Columns
                                    </Typography>
                                    
                                    {viz.metrics.map((metricId: string) => {
                                      const metric = [...metrics, ...calculatedMetrics].find(m => m.id === metricId);
                                      const columnSetting = viz.settings?.columns?.[metricId] || {};
                                      
                                      return metric ? (
                                        <Box key={metricId} sx={{ mb: 2 }}>
                                          <Typography variant="body2" fontWeight="bold">
                                            {metric.name}
                                          </Typography>
                                          <Grid container spacing={2} alignItems="center">
                                            <Grid item xs={6} md={3}>
                                              <TextField
                                                fullWidth
                                                label="Display Name"
                                                size="small"
                                                value={columnSetting.displayName || metric.name}
                                                onChange={(e) => {
                                                  const updatedColumns = {
                                                    ...viz.settings?.columns,
                                                    [metricId]: {
                                                      ...columnSetting,
                                                      displayName: e.target.value
                                                    }
                                                  };
                                                  handleUpdateVisualization(viz.id, { 
                                                    settings: { 
                                                      ...viz.settings, 
                                                      columns: updatedColumns
                                                    } 
                                                  });
                                                }}
                                              />
                                            </Grid>
                                            <Grid item xs={6} md={3}>
                                              <TextField
                                                select
                                                fullWidth
                                                label="Format"
                                                size="small"
                                                value={columnSetting.format || 'number'}
                                                onChange={(e) => {
                                                  const updatedColumns = {
                                                    ...viz.settings?.columns,
                                                    [metricId]: {
                                                      ...columnSetting,
                                                      format: e.target.value
                                                    }
                                                  };
                                                  handleUpdateVisualization(viz.id, { 
                                                    settings: { 
                                                      ...viz.settings, 
                                                      columns: updatedColumns
                                                    } 
                                                  });
                                                }}
                                              >
                                                <MenuItem value="number">Number</MenuItem>
                                                <MenuItem value="currency">Currency</MenuItem>
                                                <MenuItem value="percentage">Percentage</MenuItem>
                                                <MenuItem value="decimal">Decimal (2 places)</MenuItem>
                                              </TextField>
                                            </Grid>
                                            <Grid item xs={6} md={3}>
                                              <TextField
                                                select
                                                fullWidth
                                                label="Alignment"
                                                size="small"
                                                value={columnSetting.align || 'right'}
                                                onChange={(e) => {
                                                  const updatedColumns = {
                                                    ...viz.settings?.columns,
                                                    [metricId]: {
                                                      ...columnSetting,
                                                      align: e.target.value
                                                    }
                                                  };
                                                  handleUpdateVisualization(viz.id, { 
                                                    settings: { 
                                                      ...viz.settings, 
                                                      columns: updatedColumns
                                                    } 
                                                  });
                                                }}
                                              >
                                                <MenuItem value="left">Left</MenuItem>
                                                <MenuItem value="center">Center</MenuItem>
                                                <MenuItem value="right">Right</MenuItem>
                                              </TextField>
                                            </Grid>
                                            <Grid item xs={6} md={3}>
                                              <FormControlLabel
                                                control={
                                                  <Switch
                                                    checked={columnSetting.sortable !== false}
                                                    onChange={(e) => {
                                                      const updatedColumns = {
                                                        ...viz.settings?.columns,
                                                        [metricId]: {
                                                          ...columnSetting,
                                                          sortable: e.target.checked
                                                        }
                                                      };
                                                      handleUpdateVisualization(viz.id, { 
                                                        settings: { 
                                                          ...viz.settings, 
                                                          columns: updatedColumns
                                                        } 
                                                      });
                                                    }}
                                                  />
                                                }
                                                label="Sortable"
                                              />
                                            </Grid>
                                          </Grid>
                                        </Box>
                                      ) : null;
                                    })}
                                  </Paper>
                                )}
                              </Grid>
                            </>
                          )}
                        </Grid>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </>
          )}
        </TabPanel>
        
        {/* Layout Tab */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="body1" paragraph>
            Layout settings allow you to arrange your visualizations on the report canvas.
          </Typography>
          
          {reportBuilderVisualizations.length === 0 ? (
            <Alert severity="info">
              Add visualizations in the Visualizations tab before configuring the layout.
            </Alert>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Layout editor would be displayed here in a complete implementation.
            </Typography>
          )}
        </TabPanel>
        
        {/* Sharing Tab */}
        <TabPanel value={tabValue} index={3}>
          <Typography variant="body1" paragraph>
            Configure sharing settings for your report.
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={selectedReport?.isPublic || false}
                    onChange={(e) => {
                      if (selectedReport) {
                        // This would be handled by a proper action in a real implementation
                        // For now, we'll just show it's editable
                        console.log(`Setting report public: ${e.target.checked}`);
                      }
                    }}
                    name="isPublic"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {selectedReport?.isPublic ? (
                      <PublicIcon fontSize="small" sx={{ mr: 1 }} />
                    ) : (
                      <LockIcon fontSize="small" sx={{ mr: 1 }} />
                    )}
                    {selectedReport?.isPublic ? 'Public' : 'Private'} Report
                  </Box>
                }
              />
              <Typography variant="body2" color="text.secondary" sx={{ pl: 4 }}>
                {selectedReport?.isPublic 
                  ? 'Public reports can be viewed by anyone with the link.'
                  : 'Private reports can only be viewed by you and people you share with.'}
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Share with specific users
              </Typography>
              <Typography variant="body2" color="text.secondary">
                User permissions management would be displayed here in a complete implementation.
              </Typography>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
      
      {/* Discard Changes Dialog */}
      <Dialog
        open={discardDialogOpen}
        onClose={handleCloseDiscardDialog}
      >
        <DialogTitle>Discard Changes?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to discard your changes? Any unsaved changes will be lost.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDiscardDialog}>Cancel</Button>
          <Button onClick={handleDiscard} color="error">
            Discard
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Save Success Dialog */}
      <Dialog
        open={saveDialogOpen}
        onClose={handleCloseSaveDialog}
      >
        <DialogTitle>Report Saved</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            Your report has been saved successfully.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSaveDialog} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ReportBuilderPage;
