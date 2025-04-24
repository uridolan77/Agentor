import React, { useEffect, useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  Paper,
  Card,
  CardContent,
  CardActionArea,
  Button,
  CircularProgress,
  Divider,
  useTheme
} from '@mui/material';
import { Add as AddIcon, Collections as CollectionsIcon, BarChart as ChartIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import useReportStore from '../store/reportStore';
import { useAuth } from '../contexts/AuthContext';

const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [recentReportsLoading, setRecentReportsLoading] = useState(true);
  const [favoriteReportsLoading, setFavoriteReportsLoading] = useState(true);

  const {
    reports,
    fetchReports,
    resetReportBuilder,
    loadingReports
  } = useReportStore();

  const favoriteReports = reports.filter(report => report.isFavorite);
  const recentReports = [...reports].sort((a, b) =>
    new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  ).slice(0, 5);

  useEffect(() => {
    const loadReports = async () => {
      try {
        setRecentReportsLoading(true);
        await fetchReports();
        setRecentReportsLoading(false);
      } catch (error) {
        console.error('Error fetching reports:', error);
        setRecentReportsLoading(false);
      }
    };

    const loadFavoriteReports = async () => {
      try {
        setFavoriteReportsLoading(true);
        await fetchReports({ isFavorite: true });
        setFavoriteReportsLoading(false);
      } catch (error) {
        console.error('Error fetching favorite reports:', error);
        setFavoriteReportsLoading(false);
      }
    };

    loadReports();
    loadFavoriteReports();
  }, [fetchReports]);

  const handleCreateNewReport = () => {
    resetReportBuilder();
    navigate('/reports/builder');
  };

  const handleViewReport = (reportId: string) => {
    navigate(`/reports/${reportId}`);
  };

  const handleViewAllReports = () => {
    navigate('/reports');
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(date);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome, {user?.full_name || user?.username || 'User'}
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Access and manage your reports, or create new visualizations
        </Typography>
      </Box>

      {/* Quick Actions */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card
            onClick={handleCreateNewReport}
            sx={{
              height: '100%',
              transition: 'all 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[8]
              }
            }}
          >
            <CardActionArea sx={{ height: '100%' }}>
              <CardContent sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3
              }}>
                <AddIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" component="div">
                  Create New Report
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
                  Build a new report with customizable visualizations
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            onClick={handleViewAllReports}
            sx={{
              height: '100%',
              transition: 'all 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[8]
              }
            }}
          >
            <CardActionArea sx={{ height: '100%' }}>
              <CardContent sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3
              }}>
                <CollectionsIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" component="div">
                  View All Reports
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
                  Browse and manage all your reports
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card
            onClick={() => navigate('/data-sources')}
            sx={{
              height: '100%',
              transition: 'all 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: theme.shadows[8]
              }
            }}
          >
            <CardActionArea sx={{ height: '100%' }}>
              <CardContent sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3
              }}>
                <ChartIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" component="div">
                  Manage Data Sources
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
                  Configure connections to your data
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Reports */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Recent Reports</Typography>
          <Button onClick={handleViewAllReports}>View All</Button>
        </Box>

        <Divider sx={{ mb: 2 }} />

        {recentReportsLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : recentReports.length === 0 ? (
          <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', p: 3 }}>
            No reports found. Create your first report to get started.
          </Typography>
        ) : (
          <Grid container spacing={2}>
            {recentReports.map((report) => (
              <Grid item xs={12} md={6} key={report.id}>
                <Card
                  onClick={() => handleViewReport(report.id)}
                  sx={{
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: theme.shadows[4]
                    }
                  }}
                >
                  <CardActionArea>
                    <CardContent>
                      <Typography variant="subtitle1" noWrap>{report.name}</Typography>
                      <Typography variant="body2" color="text.secondary" noWrap sx={{ mb: 1 }}>
                        {report.description || 'No description'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Last updated: {formatDate(report.updatedAt)}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>

      {/* Favorite Reports */}
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Favorite Reports</Typography>
          <Button onClick={handleViewAllReports}>View All</Button>
        </Box>

        <Divider sx={{ mb: 2 }} />

        {favoriteReportsLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : favoriteReports.length === 0 ? (
          <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', p: 3 }}>
            No favorite reports found. Mark reports as favorites to see them here.
          </Typography>
        ) : (
          <Grid container spacing={2}>
            {favoriteReports.map((report) => (
              <Grid item xs={12} md={6} key={report.id}>
                <Card
                  onClick={() => handleViewReport(report.id)}
                  sx={{
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: theme.shadows[4]
                    }
                  }}
                >
                  <CardActionArea>
                    <CardContent>
                      <Typography variant="subtitle1" noWrap>{report.name}</Typography>
                      <Typography variant="body2" color="text.secondary" noWrap sx={{ mb: 1 }}>
                        {report.description || 'No description'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Last updated: {formatDate(report.updatedAt)}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Container>
  );
};

export default DashboardPage;