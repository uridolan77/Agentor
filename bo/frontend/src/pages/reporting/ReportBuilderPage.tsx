import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  TextField,
  Tabs,
  Tab,
  IconButton,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Save as SaveIcon,
  Preview as PreviewIcon,
  ViewColumn as ViewColumnIcon,
  Functions as FunctionsIcon,
  BarChart as BarChartIcon,
  TableChart as TableChartIcon,
  PieChart as PieChartIcon,
  ShowChart as LineChartIcon
} from '@mui/icons-material';

// Import API and types
import * as dataObjectsApi from '../../components/data-object-canvas/data-objects-api';
import { DataModel, TableSchema } from '../../components/data-object-canvas/data-objects-api';
import useReportStore from '../../store/reportStore';

// Import custom components
import DataModelSelector from '../../components/reporting/DataModelSelector';
import FieldSelector, { Field } from '../../components/reporting/FieldSelector';

// TabPanel component for the tabs
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`report-builder-tabpanel-${index}`}
      aria-labelledby={`report-builder-tab-${index}`}
      {...other}
      style={{ padding: '16px' }}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

// Steps for the report building process
const steps = ['Select Data Model', 'Choose Fields', 'Configure Visualizations', 'Preview & Save'];

// Main component
const ReportBuilderPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const reportId = searchParams.get('id');
  const modelId = searchParams.get('modelId');

  // State variables
  const [activeStep, setActiveStep] = useState(0);
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [processingModelId, setProcessingModelId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dataModels, setDataModels] = useState<DataModel[]>([]);
  const [selectedDataModel, setSelectedDataModel] = useState<DataModel | null>(null);
  const [tableSchemas, setTableSchemas] = useState<TableSchema[]>([]);
  const [selectedDimensions, setSelectedDimensions] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [reportNameValue, setReportNameValue] = useState('New Report');
  const [reportDescValue, setReportDescValue] = useState('');
  const [saving, setSaving] = useState(false);

  // Get data sources from the store
  const {
    dataSources,
    fetchDataSources,
    loadingDataSources,
    createReport,
    updateReport,
    setReportName,
    setReportDescription,
    selectDataSource,
    resetReportBuilder,
    addField,
    addVisualization
  } = useReportStore();

  // Load data sources and data models on component mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        await fetchDataSources();

        // If we have a modelId in the URL, load that model
        if (modelId) {
          setProcessingModelId(modelId);
          await loadDataModel(modelId);
          setActiveStep(1); // Move to the next step
          setProcessingModelId(null);
        } else if (dataSources.length > 0) {
          // Otherwise, load all data models for the first data source
          const models = await dataObjectsApi.getDataModels(dataSources[0].id);
          setDataModels(models);
        }

        setLoading(false);
      } catch (err) {
        console.error('Error loading initial data:', err);
        setError('Failed to load initial data. Please try again.');
        setLoading(false);
      }
    };

    loadInitialData();
  }, [fetchDataSources, modelId]);

  // Load a specific data model
  const loadDataModel = async (modelId: string) => {
    try {
      setLoading(true);
      const model = await dataObjectsApi.getDataModel(modelId);
      setSelectedDataModel(model);

      // Load table schemas for this data model
      if (model.dataSourceId) {
        const schemas = await dataObjectsApi.getDatabaseSchema(model.dataSourceId);

        // Filter schemas to only include tables in the data model
        const filteredSchemas = schemas.filter(schema =>
          model.tables.includes(schema.name)
        );

        setTableSchemas(filteredSchemas);
      }

      setLoading(false);
    } catch (err) {
      console.error('Error loading data model:', err);
      setError('Failed to load data model. Please try again.');
      setLoading(false);
    }
  };

  // Handle data model selection
  const handleDataModelSelect = async (modelId: string) => {
    try {
      setProcessingModelId(modelId);
      await loadDataModel(modelId);
      setActiveStep(1); // Move to the next step
    } finally {
      setProcessingModelId(null);
    }
  };

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // These handlers are now replaced by handleDimensionToggle and handleMetricToggle

  // Handle next step
  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  // Handle back step
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  // Handle save report
  const handleSaveReport = async () => {
    try {
      setSaving(true);

      // Set up the report in the store
      setReportName(reportNameValue);
      setReportDescription(reportDescValue);
      selectDataSource(selectedDataModel?.dataSourceId || '');

      // Clear existing fields and add new ones
      resetReportBuilder();

      // Add fields to the store
      selectedDimensions.forEach(dim => {
        addField({
          id: dim,
          type: 'dimension',
          name: dim
        });
      });

      selectedMetrics.forEach(metric => {
        addField({
          id: metric,
          type: 'metric',
          name: metric
        });
      });

      // Add a default visualization
      addVisualization({
        id: '1',
        type: 'table',
        title: 'Data Table',
        dimensions: selectedDimensions,
        metrics: selectedMetrics,
        settings: {}
      });

      // Create or update the report
      if (reportId) {
        await updateReport();
      } else {
        await createReport();
      }

      setSaving(false);
      navigate('/reporting/reports');
    } catch (err) {
      console.error('Error saving report:', err);
      setError('Failed to save report. Please try again.');
      setSaving(false);
    }
  };

  // Get all dimensions from the table schemas
  const getDimensions = (): Field[] => {
    const dimensions: Field[] = [];

    tableSchemas.forEach(table => {
      table.columns.forEach(column => {
        // Consider non-numeric columns as dimensions
        if (!['int', 'float', 'double', 'decimal', 'number'].includes(column.type.toLowerCase())) {
          dimensions.push({
            id: `${table.name}.${column.name}`,
            name: column.name,
            type: column.type,
            table: table.name,
            description: `${column.type} column from ${table.name} table`
          });
        }
      });
    });

    return dimensions;
  };

  // Get all metrics from the table schemas
  const getMetrics = (): Field[] => {
    const metrics: Field[] = [];

    tableSchemas.forEach(table => {
      table.columns.forEach(column => {
        // Consider numeric columns as metrics
        if (['int', 'float', 'double', 'decimal', 'number'].includes(column.type.toLowerCase())) {
          metrics.push({
            id: `${table.name}.${column.name}`,
            name: column.name,
            type: column.type,
            table: table.name,
            description: `${column.type} column from ${table.name} table`
          });
        }
      });
    });

    return metrics;
  };

  // Handle dimension toggle
  const handleDimensionToggle = (dimensionId: string) => {
    if (selectedDimensions.includes(dimensionId)) {
      setSelectedDimensions(selectedDimensions.filter(id => id !== dimensionId));
    } else {
      setSelectedDimensions([...selectedDimensions, dimensionId]);
    }
  };

  // Handle metric toggle
  const handleMetricToggle = (metricId: string) => {
    if (selectedMetrics.includes(metricId)) {
      setSelectedMetrics(selectedMetrics.filter(id => id !== metricId));
    } else {
      setSelectedMetrics([...selectedMetrics, metricId]);
    }
  };

  // Render the step content based on the active step
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select a Data Model
            </Typography>

            <DataModelSelector
              dataModels={dataModels}
              loading={loading}
              onSelect={handleDataModelSelect}
              selectedModelId={selectedDataModel?.id}
              processingModelId={processingModelId}
            />
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Choose Fields for Your Report
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <FieldSelector
                dimensions={getDimensions()}
                metrics={getMetrics()}
                selectedDimensions={selectedDimensions}
                selectedMetrics={selectedMetrics}
                onDimensionToggle={handleDimensionToggle}
                onMetricToggle={handleMetricToggle}
              />
            </Paper>

            <Typography variant="subtitle1" gutterBottom>
              Selected Fields Preview
            </Typography>

            <Paper sx={{ p: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2">Dimensions:</Typography>
                  <List dense>
                    {selectedDimensions.map((dim) => (
                      <ListItem key={dim}>
                        <ListItemIcon>
                          <ViewColumnIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={dim} />
                      </ListItem>
                    ))}
                    {selectedDimensions.length === 0 && (
                      <ListItem>
                        <ListItemText secondary="No dimensions selected" />
                      </ListItem>
                    )}
                  </List>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2">Metrics:</Typography>
                  <List dense>
                    {selectedMetrics.map((metric) => (
                      <ListItem key={metric}>
                        <ListItemIcon>
                          <FunctionsIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={metric} />
                      </ListItem>
                    ))}
                    {selectedMetrics.length === 0 && (
                      <ListItem>
                        <ListItemText secondary="No metrics selected" />
                      </ListItem>
                    )}
                  </List>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Configure Visualizations
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="visualization tabs">
                <Tab label="Table" icon={<TableChartIcon />} />
                <Tab label="Bar Chart" icon={<BarChartIcon />} />
                <Tab label="Line Chart" icon={<LineChartIcon />} />
                <Tab label="Pie Chart" icon={<PieChartIcon />} />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <Typography variant="subtitle1" gutterBottom>
                  Table Configuration
                </Typography>

                {selectedDimensions.length === 0 && selectedMetrics.length === 0 ? (
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    Please go back and select dimensions and metrics first.
                  </Alert>
                ) : (
                  <>
                    <Alert severity="info" sx={{ mb: 3 }}>
                      The table will display all selected dimensions and metrics.
                    </Alert>

                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>Selected Dimensions:</Typography>
                      <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                        {selectedDimensions.map((dim) => (
                          <Chip key={dim} label={dim} sx={{ m: 0.5 }} />
                        ))}
                      </Paper>

                      <Typography variant="subtitle2" gutterBottom>Selected Metrics:</Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                        {selectedMetrics.map((metric) => (
                          <Chip key={metric} label={metric} color="primary" sx={{ m: 0.5 }} />
                        ))}
                      </Paper>
                    </Box>

                    <Box sx={{ mt: 3, p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>Table Preview</Typography>
                      <Box sx={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              {selectedDimensions.map((dim) => (
                                <th key={dim} style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                                  {dim.split('.')[1]}
                                </th>
                              ))}
                              {selectedMetrics.map((metric) => (
                                <th key={metric} style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #ddd' }}>
                                  {metric.split('.')[1]}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              {selectedDimensions.map((dim) => (
                                <td key={dim} style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                                  Sample data
                                </td>
                              ))}
                              {selectedMetrics.map((metric) => (
                                <td key={metric} style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #ddd' }}>
                                  123
                                </td>
                              ))}
                            </tr>
                            <tr>
                              {selectedDimensions.map((dim) => (
                                <td key={dim} style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                                  Sample data 2
                                </td>
                              ))}
                              {selectedMetrics.map((metric) => (
                                <td key={metric} style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid #ddd' }}>
                                  456
                                </td>
                              ))}
                            </tr>
                          </tbody>
                        </table>
                      </Box>
                    </Box>
                  </>
                )}
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <Typography variant="subtitle1" gutterBottom>
                  Bar Chart Configuration
                </Typography>

                {selectedDimensions.length === 0 || selectedMetrics.length === 0 ? (
                  <Alert severity="warning">
                    Please go back and select at least one dimension and one metric.
                  </Alert>
                ) : (
                  <>
                    <Alert severity="info" sx={{ mb: 3 }}>
                      Bar charts work best with categorical dimensions and numeric metrics.
                    </Alert>

                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>X-Axis (Dimension):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                        {selectedDimensions.length > 0 && (
                          <Chip label={selectedDimensions[0]} sx={{ m: 0.5 }} />
                        )}
                      </Paper>

                      <Typography variant="subtitle2" gutterBottom>Y-Axis (Metrics):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                        {selectedMetrics.map((metric) => (
                          <Chip key={metric} label={metric} color="primary" sx={{ m: 0.5 }} />
                        ))}
                      </Paper>
                    </Box>

                    <Box sx={{ mt: 3, p: 2, border: 1, borderColor: 'divider', borderRadius: 1, height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Bar chart preview will be shown here
                      </Typography>
                    </Box>
                  </>
                )}
              </TabPanel>

              <TabPanel value={tabValue} index={2}>
                <Typography variant="subtitle1" gutterBottom>
                  Line Chart Configuration
                </Typography>

                {selectedDimensions.length === 0 || selectedMetrics.length === 0 ? (
                  <Alert severity="warning">
                    Please go back and select at least one dimension and one metric.
                  </Alert>
                ) : (
                  <>
                    <Alert severity="info" sx={{ mb: 3 }}>
                      Line charts work best with time-based dimensions and numeric metrics.
                    </Alert>

                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>X-Axis (Time Dimension):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                        {selectedDimensions.length > 0 && (
                          <Chip label={selectedDimensions[0]} sx={{ m: 0.5 }} />
                        )}
                      </Paper>

                      <Typography variant="subtitle2" gutterBottom>Y-Axis (Metrics):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                        {selectedMetrics.map((metric) => (
                          <Chip key={metric} label={metric} color="primary" sx={{ m: 0.5 }} />
                        ))}
                      </Paper>
                    </Box>

                    <Box sx={{ mt: 3, p: 2, border: 1, borderColor: 'divider', borderRadius: 1, height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Line chart preview will be shown here
                      </Typography>
                    </Box>
                  </>
                )}
              </TabPanel>

              <TabPanel value={tabValue} index={3}>
                <Typography variant="subtitle1" gutterBottom>
                  Pie Chart Configuration
                </Typography>

                {selectedDimensions.length === 0 || selectedMetrics.length === 0 ? (
                  <Alert severity="warning">
                    Please go back and select at least one dimension and one metric.
                  </Alert>
                ) : (
                  <>
                    <Alert severity="info" sx={{ mb: 3 }}>
                      Pie charts work best with a single dimension and a single metric.
                    </Alert>

                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" gutterBottom>Segments (Dimension):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                        {selectedDimensions.length > 0 && (
                          <Chip label={selectedDimensions[0]} sx={{ m: 0.5 }} />
                        )}
                      </Paper>

                      <Typography variant="subtitle2" gutterBottom>Values (Metric):</Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                        {selectedMetrics.length > 0 && (
                          <Chip label={selectedMetrics[0]} color="primary" sx={{ m: 0.5 }} />
                        )}
                      </Paper>
                    </Box>

                    <Box sx={{ mt: 3, p: 2, border: 1, borderColor: 'divider', borderRadius: 1, height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Pie chart preview will be shown here
                      </Typography>
                    </Box>
                  </>
                )}
              </TabPanel>
            </Paper>
          </Box>
        );

      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Preview & Save
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Report Name"
                    value={reportNameValue}
                    onChange={(e) => setReportNameValue(e.target.value)}
                    required
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Description"
                    value={reportDescValue}
                    onChange={(e) => setReportDescValue(e.target.value)}
                    multiline
                    rows={2}
                  />
                </Grid>
              </Grid>
            </Paper>

            <Typography variant="subtitle1" gutterBottom>
              Report Summary
            </Typography>

            <Paper sx={{ p: 3 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2">Data Model:</Typography>
                  <Typography variant="body2">{selectedDataModel?.name}</Typography>

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Dimensions ({selectedDimensions.length}):</Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {selectedDimensions.map((dim) => (
                        <li key={dim}>
                          <Typography variant="body2">{dim}</Typography>
                        </li>
                      ))}
                    </Box>
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2">Visualizations:</Typography>
                  <Box component="ul" sx={{ pl: 2 }}>
                    <li>
                      <Typography variant="body2">Table View</Typography>
                    </li>
                    {tabValue === 1 && (
                      <li>
                        <Typography variant="body2">Bar Chart</Typography>
                      </li>
                    )}
                    {tabValue === 2 && (
                      <li>
                        <Typography variant="body2">Line Chart</Typography>
                      </li>
                    )}
                    {tabValue === 3 && (
                      <li>
                        <Typography variant="body2">Pie Chart</Typography>
                      </li>
                    )}
                  </Box>

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Metrics ({selectedMetrics.length}):</Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {selectedMetrics.map((metric) => (
                        <li key={metric}>
                          <Typography variant="body2">{metric}</Typography>
                        </li>
                      ))}
                    </Box>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        );

      default:
        return 'Unknown step';
    }
  };

  // Show loading state
  if (loadingDataSources) {
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

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header with back button and actions */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton component={Link} to="/reporting/reports" sx={{ mr: 1 }}>
            <ArrowBackIcon />
          </IconButton>
          <Typography variant="h4" component="h1">
            {reportId ? 'Edit Report' : 'New Report'}
          </Typography>
        </Box>

        <Box>
          {activeStep === steps.length - 1 && (
            <>
              <Button
                variant="outlined"
                startIcon={<PreviewIcon />}
                sx={{ mr: 1 }}
              >
                Preview
              </Button>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={handleSaveReport}
                disabled={saving || !reportNameValue || selectedDimensions.length === 0}
              >
                {saving ? <CircularProgress size={24} /> : 'Save Report'}
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stepper */}
      <Paper sx={{ mb: 4, p: 3 }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Step content */}
      <Paper sx={{ p: 3, mb: 4 }}>
        {getStepContent(activeStep)}
      </Paper>

      {/* Navigation buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
        <Button
          variant="outlined"
          onClick={handleBack}
          disabled={activeStep === 0}
        >
          Back
        </Button>

        <Button
          variant="contained"
          onClick={handleNext}
          disabled={
            activeStep === steps.length - 1 ||
            (activeStep === 0 && !selectedDataModel) ||
            (activeStep === 1 && selectedDimensions.length === 0)
          }
        >
          Next
        </Button>
      </Box>
    </Container>
  );
};

export default ReportBuilderPage;
