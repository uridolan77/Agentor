import React, { useState, useRef, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  CircularProgress,
  Tooltip,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  MoreVert as MoreIcon,
  Refresh as RefreshIcon,
  Share as ShareIcon,
  GetApp as DownloadIcon,
  Print as PrintIcon,
  PictureAsPdf as PdfIcon,
  FilterList as FilterIcon,
  TableChart as CsvIcon
} from '@mui/icons-material';
import { useReactToPrint } from 'react-to-print';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { Parser } from 'json2csv';

import reportingAPI from '../../api/reportingApi';
import { Report, ReportData, ReportElement, FilterCondition } from '../../types/reporting/reportTypes';
import ChartRenderer from './ChartRenderer';

// Styled components
const ViewerContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  backgroundColor: theme.palette.background.default,
  minHeight: '100vh',
}));

const ReportHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing(3),
}));

const ReportTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
}));

const ReportDescription = styled(Typography)(({ theme }) => ({
  color: theme.palette.text.secondary,
  marginTop: theme.spacing(1),
}));

const ReportActions = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
}));

const ElementCard = styled(Paper)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[2],
}));

const ElementHeader = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  paddingBottom: 0,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
}));

const ElementTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 500,
}));

const ElementContent = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  flexGrow: 1,
  overflow: 'auto',
}));

const LoadingContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '50vh',
}));

const ErrorContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  textAlign: 'center',
  color: theme.palette.error.main,
}));

// Helper function to get grid size based on element type
const getGridSize = (element: ReportElement) => {
  const { visualization } = element;
  
  switch (visualization.type) {
    case 'table':
      return { xs: 12, sm: 12, md: 12 };
    case 'card':
      return { xs: 12, sm: 6, md: 3 };
    case 'pie_chart':
      return { xs: 12, sm: 6, md: 6 };
    default:
      return { xs: 12, sm: 12, md: 6 };
  }
};

interface ReportViewerProps {
  reportId: string;
  onBack?: () => void;
}

const ReportViewer: React.FC<ReportViewerProps> = ({ reportId, onBack }) => {
  const [report, setReport] = useState<Report | null>(null);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [filterDialogOpen, setFilterDialogOpen] = useState<boolean>(false);
  const [filters, setFilters] = useState<Record<string, any>>({});
  
  const reportRef = useRef<HTMLDivElement>(null);
  
  // Load report data
  useEffect(() => {
    const loadReport = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const reportData = await reportingAPI.getReport(reportId);
        setReport(reportData);
        
        // Load report data
        await generateReportData();
      } catch (error) {
        console.error('Error loading report:', error);
        setError('Failed to load report. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    loadReport();
  }, [reportId]);
  
  // Generate report data
  const generateReportData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await reportingAPI.generateReportData({
        reportId,
        parameters: filters
      });
      
      setReportData(data);
    } catch (error) {
      console.error('Error generating report data:', error);
      setError('Failed to generate report data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle menu open/close
  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setMenuAnchor(null);
  };
  
  // Handle print
  const handlePrint = useReactToPrint({
    content: () => reportRef.current,
    documentTitle: report?.name || 'Report',
  });
  
  // Handle PDF export
  const handleExportPDF = async () => {
    if (!reportRef.current) return;
    
    try {
      const canvas = await html2canvas(reportRef.current);
      const imgData = canvas.toDataURL('image/png');
      
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });
      
      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
      pdf.save(`${report?.name || 'report'}.pdf`);
    } catch (error) {
      console.error('Error exporting PDF:', error);
      setError('Failed to export PDF. Please try again later.');
    }
  };
  
  // Handle CSV export
  const handleExportCSV = () => {
    if (!report || !reportData) return;
    
    try {
      // Prepare fields for CSV
      const fields: string[] = [];
      
      report.elements.forEach((element: ReportElement) => {
        element.dimensions.forEach((dimId: string) => {
          const field = `dimension_${dimId}`;
          if (!fields.includes(field)) fields.push(field);
        });
        
        element.metrics.forEach((metricId: string) => {
          const field = `metric_${metricId}`;
          if (!fields.includes(field)) fields.push(field);
        });
      });
      
      // Create CSV
      const parser = new Parser({ fields });
      const csv = parser.parse(reportData.data);
      
      // Download CSV
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `${report.name || 'report'}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error exporting CSV:', error);
      setError('Failed to export CSV. Please try again later.');
    }
  };
  
  // Handle filter dialog
  const handleFilterDialogOpen = () => {
    setFilterDialogOpen(true);
  };
  
  const handleFilterDialogClose = () => {
    setFilterDialogOpen(false);
  };
  
  const handleFilterChange = (key: string, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  const handleApplyFilters = () => {
    generateReportData();
    setFilterDialogOpen(false);
  };
  
  // Render loading state
  if (isLoading && !report) {
    return (
      <LoadingContainer>
        <CircularProgress size={40} />
        <Typography variant="body1" color="textSecondary" sx={{ mt: 2 }}>
          Loading report...
        </Typography>
      </LoadingContainer>
    );
  }
  
  // Render error state
  if (error && !report) {
    return (
      <ErrorContainer>
        <Typography variant="h6" color="error">
          Error
        </Typography>
        <Typography variant="body1">
          {error}
        </Typography>
        <Button variant="outlined" color="primary" onClick={onBack} sx={{ mt: 2 }}>
          Back to Reports
        </Button>
      </ErrorContainer>
    );
  }
  
  // Render report
  return (
    <ViewerContainer>
      {/* Report Header */}
      <ReportHeader>
        <Box>
          <ReportTitle variant="h4">
            {report?.name || 'Report'}
          </ReportTitle>
          {report?.description && (
            <ReportDescription variant="body1">
              {report.description}
            </ReportDescription>
          )}
        </Box>
        
        <ReportActions>
          <Button
            variant="outlined"
            startIcon={<FilterIcon />}
            onClick={handleFilterDialogOpen}
          >
            Filters
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={generateReportData}
          >
            Refresh
          </Button>
          
          <IconButton onClick={handleMenuOpen}>
            <MoreIcon />
          </IconButton>
          
          <Menu
            anchorEl={menuAnchor}
            open={Boolean(menuAnchor)}
            onClose={handleMenuClose}
          >
            <MenuItem onClick={handlePrint}>
              <PrintIcon fontSize="small" sx={{ mr: 1 }} />
              Print
            </MenuItem>
            <MenuItem onClick={handleExportPDF}>
              <PdfIcon fontSize="small" sx={{ mr: 1 }} />
              Export as PDF
            </MenuItem>
            <MenuItem onClick={handleExportCSV}>
              <CsvIcon fontSize="small" sx={{ mr: 1 }} />
              Export as CSV
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleMenuClose}>
              <ShareIcon fontSize="small" sx={{ mr: 1 }} />
              Share
            </MenuItem>
          </Menu>
        </ReportActions>
      </ReportHeader>
      
      {/* Report Content */}
      <div ref={reportRef}>
        {isLoading && (
          <LoadingContainer>
            <CircularProgress size={40} />
            <Typography variant="body1" color="textSecondary" sx={{ mt: 2 }}>
              Generating report data...
            </Typography>
          </LoadingContainer>
        )}
        
        {error && !isLoading && (
          <ErrorContainer>
            <Typography variant="h6" color="error">
              Error
            </Typography>
            <Typography variant="body1">
              {error}
            </Typography>
            <Button variant="outlined" color="primary" onClick={generateReportData} sx={{ mt: 2 }}>
              Try Again
            </Button>
          </ErrorContainer>
        )}
        
        {!isLoading && !error && report && (
          <>
            {/* Report Elements */}
            <Grid container spacing={3}>
              {report.elements.map((element: ReportElement) => {
                const gridSize = getGridSize(element);
                
                return (
                  <Grid item xs={gridSize.xs} sm={gridSize.sm} md={gridSize.md} key={element.id}>
                    <ElementCard>
                      <ElementHeader>
                        <ElementTitle variant="h6">
                          {element.name}
                        </ElementTitle>
                      </ElementHeader>
                      
                      <ElementContent>
                        <ChartRenderer 
                          element={element}
                          data={reportData?.data || []}
                        />
                      </ElementContent>
                    </ElementCard>
                  </Grid>
                );
              })}
            </Grid>
          </>
        )}
      </div>
      
      {/* Filter Dialog */}
      <Dialog open={filterDialogOpen} onClose={handleFilterDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>Filter Report</DialogTitle>
        <DialogContent>
          {report?.globalFilters?.map((filter: FilterCondition) => (
            <TextField
              key={filter.id}
              label={filter.fieldId}
              fullWidth
              margin="normal"
              value={filters[filter.fieldId] || ''}
              onChange={(e) => handleFilterChange(filter.fieldId, e.target.value)}
            />
          ))}
          
          {(!report?.globalFilters || report.globalFilters.length === 0) && (
            <Typography variant="body1" color="textSecondary">
              No filters available for this report.
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleFilterDialogClose}>
            Cancel
          </Button>
          <Button onClick={handleApplyFilters} variant="contained" color="primary">
            Apply Filters
          </Button>
        </DialogActions>
      </Dialog>
    </ViewerContainer>
  );
};

export default ReportViewer;
