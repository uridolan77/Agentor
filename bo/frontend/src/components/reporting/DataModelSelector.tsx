import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  CardActionArea,
  Typography,
  Chip,
  Skeleton,
  Button,
  CircularProgress,
  Backdrop,
  Alert
} from '@mui/material';
import { Link } from 'react-router-dom';
import { DataModel } from '../../components/data-object-canvas/data-objects-api';

interface DataModelSelectorProps {
  dataModels: DataModel[];
  loading: boolean;
  onSelect: (modelId: string) => void;
  selectedModelId?: string;
  processingModelId?: string;
}

const DataModelSelector: React.FC<DataModelSelectorProps> = ({
  dataModels,
  loading,
  onSelect,
  selectedModelId,
  processingModelId
}) => {
  if (loading) {
    return (
      <Grid container spacing={3}>
        {[1, 2, 3].map((item) => (
          <Grid item xs={12} sm={6} md={4} key={item}>
            <Card>
              <CardHeader
                title={<Skeleton variant="text" width="80%" />}
                subheader={<Skeleton variant="text" width="40%" />}
              />
              <CardContent>
                <Skeleton variant="rectangular" height={60} />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  if (dataModels.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          No Data Models Found
        </Typography>
        <Typography variant="body1" paragraph>
          You need to create a data model before you can build a report.
        </Typography>
        <Button
          variant="contained"
          component={Link}
          to="/reporting/data-canvas"
        >
          Create Data Model
        </Button>
      </Box>
    );
  }

  return (
    <>
      {processingModelId && (
        <Backdrop
          sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1, flexDirection: 'column', gap: 2 }}
          open={!!processingModelId}
        >
          <CircularProgress color="inherit" />
          <Typography variant="h6">Loading data model...</Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This may take a moment while we prepare your report builder
          </Alert>
        </Backdrop>
      )}
      <Grid container spacing={3}>
        {dataModels.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <Card
              sx={{
                height: '100%',
                border: selectedModelId === model.id ? 2 : 0,
                borderColor: 'primary.main',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4
                }
              }}
            >
              <CardActionArea
                onClick={() => onSelect(model.id)}
                disabled={!!processingModelId}
                sx={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'stretch' }}
              >
              <CardHeader
                title={model.name}
                subheader={`Created: ${new Date(model.createdAt).toLocaleDateString()}`}
                titleTypographyProps={{ variant: 'h6' }}
                subheaderTypographyProps={{ variant: 'caption' }}
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {model.tables.length} tables
                </Typography>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {model.relationships.length} relationships
                </Typography>

                <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {model.tables.slice(0, 5).map((table) => (
                    <Chip
                      key={table}
                      label={table}
                      size="small"
                      variant="outlined"
                      sx={{ mb: 0.5 }}
                    />
                  ))}
                  {model.tables.length > 5 && (
                    <Chip
                      label={`+${model.tables.length - 5} more`}
                      size="small"
                      variant="outlined"
                    />
                  )}
                </Box>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      ))}
    </Grid>
    </>
  );
};

export default DataModelSelector;
