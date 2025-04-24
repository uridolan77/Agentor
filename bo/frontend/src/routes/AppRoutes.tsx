import React from 'react';
import { Navigate, RouteObject, useRoutes } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

// Layout
import MainLayout from '../layouts/MainLayout';

// Pages
import LoginPage from '../pages/LoginPage';
import DashboardPage from '../pages/DashboardPage';
// These components will be implemented later
const ReportsListPage = React.lazy(() => import('../pages/ReportsListPage'));
const ReportBuilderPage = React.lazy(() => import('../pages/reporting/ReportBuilderPage'));
const ReportViewerPage = React.lazy(() => import('../pages/ReportViewerPage'));
const DataSourcesPage = React.lazy(() => import('../pages/DataSourcesPage'));
const DataCanvasPage = React.lazy(() => import('../pages/reporting/DataCanvasPage'));
const DataModelsPage = React.lazy(() => import('../pages/reporting/DataModelsPage'));
const SettingsPage = React.lazy(() => import('../pages/SettingsPage'));
const ProfilePage = React.lazy(() => import('../pages/ProfilePage'));
const NotFoundPage = React.lazy(() => import('../pages/NotFoundPage'));
// Workflow pages
const WorkflowsPage = React.lazy(() => import('../pages/workflows/WorkflowsPage'));
const WorkflowDetailPage = React.lazy(() => import('../pages/workflows/WorkflowDetailPage'));
// Agent pages
const AgentsPage = React.lazy(() => import('../pages/agents/AgentsPage'));
const AgentDetailPage = React.lazy(() => import('../pages/agents/AgentDetailPage'));
const CreateEditAgentPage = React.lazy(() => import('../pages/agents/CreateEditAgentPage'));
// Tool pages
const ToolsPage = React.lazy(() => import('../pages/tools/ToolsPage'));
const ToolDetailPage = React.lazy(() => import('../pages/tools/ToolDetailPage'));
const CreateEditToolPage = React.lazy(() => import('../pages/tools/CreateEditToolPage'));
// Training pages
const TrainingPage = React.lazy(() => import('../pages/training/TrainingPage'));

// Protected route component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    // You could render a loading spinner here
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return <>{children}</>;
};

// Public route that redirects to dashboard if already authenticated
interface PublicRouteProps {
  children: React.ReactNode;
}

const PublicRoute: React.FC<PublicRouteProps> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    // You could render a loading spinner here
    return <div>Loading...</div>;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }

  return <>{children}</>;
};

const AppRoutes: React.FC = () => {
  const routes: RouteObject[] = [
    {
      path: '/login',
      element: (
        <PublicRoute>
          <LoginPage />
        </PublicRoute>
      ),
    },
    {
      path: '/',
      element: (
        <ProtectedRoute>
          <MainLayout />
        </ProtectedRoute>
      ),
      children: [
        {
          path: '',
          element: <Navigate to="/dashboard" />,
        },
        {
          path: 'dashboard',
          element: <DashboardPage />,
        },
        {
          path: 'agents',
          children: [
            {
              path: '',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <AgentsPage />
                </React.Suspense>
              ),
            },
            {
              path: 'create',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <CreateEditAgentPage />
                </React.Suspense>
              ),
            },
            {
              path: 'edit/:id',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <CreateEditAgentPage />
                </React.Suspense>
              ),
            },
            {
              path: ':id',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <AgentDetailPage />
                </React.Suspense>
              ),
            },
          ],
        },
        {
          path: 'tools',
          children: [
            {
              path: '',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ToolsPage />
                </React.Suspense>
              ),
            },
            {
              path: 'create',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <CreateEditToolPage />
                </React.Suspense>
              ),
            },
            {
              path: 'edit/:id',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <CreateEditToolPage />
                </React.Suspense>
              ),
            },
            {
              path: ':id',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ToolDetailPage />
                </React.Suspense>
              ),
            },
          ],
        },
        {
          path: 'training',
          element: (
            <React.Suspense fallback={<div>Loading...</div>}>
              <TrainingPage />
            </React.Suspense>
          ),
        },
        {
          path: 'workflows',
          children: [
            {
              path: '',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <WorkflowsPage />
                </React.Suspense>
              ),
            },
            {
              path: ':id',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <WorkflowDetailPage />
                </React.Suspense>
              ),
            },
          ],
        },
        {
          path: 'reports',
          children: [
            {
              path: '',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportsListPage />
                </React.Suspense>
              ),
            },
            {
              path: 'builder',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportBuilderPage />
                </React.Suspense>
              ),
            },
            {
              path: ':reportId',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportViewerPage />
                </React.Suspense>
              ),
            },
          ],
        },
        {
          path: 'data-sources',
          element: (
            <React.Suspense fallback={<div>Loading...</div>}>
              <DataSourcesPage />
            </React.Suspense>
          ),
        },
        {
          path: 'reporting',
          children: [
            {
              path: 'data-canvas',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <DataCanvasPage />
                </React.Suspense>
              ),
            },
            {
              path: 'data-models',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <DataModelsPage />
                </React.Suspense>
              ),
            },
            {
              path: 'reports',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportsListPage />
                </React.Suspense>
              ),
            },
            {
              path: 'reports/builder',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportBuilderPage />
                </React.Suspense>
              ),
            },
            {
              path: 'reports/:reportId',
              element: (
                <React.Suspense fallback={<div>Loading...</div>}>
                  <ReportViewerPage />
                </React.Suspense>
              ),
            },
          ],
        },
        {
          path: 'settings',
          element: (
            <React.Suspense fallback={<div>Loading...</div>}>
              <SettingsPage />
            </React.Suspense>
          ),
        },
        {
          path: 'profile',
          element: (
            <React.Suspense fallback={<div>Loading...</div>}>
              <ProfilePage />
            </React.Suspense>
          ),
        },
        {
          path: '*',
          element: (
            <React.Suspense fallback={<div>Loading...</div>}>
              <NotFoundPage />
            </React.Suspense>
          ),
        },
      ],
    },
  ];

  const routing = useRoutes(routes);

  return <>{routing}</>;
};

export default AppRoutes;