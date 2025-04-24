import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Divider,
  Avatar,
  Menu,
  MenuItem,
  useTheme,
  useMediaQuery,
  Collapse
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  SmartToy as AgentIcon,
  Build as ToolIcon,
  AccountTree as WorkflowIcon,
  Cloud as LLMIcon,
  School as TrainingIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
  Person as PersonIcon,
  ExitToApp as LogoutIcon,
  ExpandLess,
  ExpandMore,
  Assessment as ReportIcon,
  Storage as DataSourceIcon,
  Edit as BuilderIcon,
  AutoGraph as AutoGraphIcon,
  Schema as SchemaIcon
} from '@mui/icons-material';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

// Drawer width
const drawerWidth = 240;

const MainLayout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { user, logout, hasPermission } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // State to track expanded menu items - auto-expand Reports menu if on a reports page
  const [expandedMenus, setExpandedMenus] = useState<Record<string, boolean>>({
    reports: location.pathname.startsWith('/reports') ||
             location.pathname.startsWith('/data-sources') ||
             location.pathname.startsWith('/reporting')
  });

  // Toggle menu expansion
  const handleMenuExpand = (menuId: string) => {
    setExpandedMenus(prev => ({
      ...prev,
      [menuId]: !prev[menuId]
    }));
  };

  // Update expanded menus when location changes
  useEffect(() => {
    const isReportsActive = location.pathname.startsWith('/reports') ||
                           location.pathname.startsWith('/data-sources') ||
                           location.pathname.startsWith('/reporting');

    setExpandedMenus(prev => ({
      ...prev,
      reports: isReportsActive
    }));
  }, [location.pathname]);

  // Menu items with permissions
  const menuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard',
      permission: null
    },
    {
      text: 'Agents',
      icon: <AgentIcon />,
      path: '/agents',
      permission: 'agent:read'
    },
    {
      text: 'Tools',
      icon: <ToolIcon />,
      path: '/tools',
      permission: 'tool:read'
    },
    {
      text: 'Workflows',
      icon: <WorkflowIcon />,
      path: '/workflows',
      permission: 'workflow:read'
    },
    {
      text: 'LLM Connections',
      icon: <LLMIcon />,
      path: '/llm',
      permission: 'llm:read'
    },
    {
      text: 'Training',
      icon: <TrainingIcon />,
      path: '/training',
      permission: 'training:read'
    },
    {
      text: 'Reports',
      icon: <ReportIcon />,
      path: null,
      permission: null,
      id: 'reports',
      children: [
        {
          text: 'Reports List',
          icon: <ReportIcon />,
          path: '/reporting/reports',
          permission: null
        },
        {
          text: 'Report Builder',
          icon: <BuilderIcon />,
          path: '/reporting/reports/builder',
          permission: null
        },
        {
          text: 'Data Sources',
          icon: <DataSourceIcon />,
          path: '/data-sources',
          permission: null
        },
        {
          text: 'Data Canvas',
          icon: <AutoGraphIcon />,
          path: '/reporting/data-canvas',
          permission: null
        },
        {
          text: 'Data Models',
          icon: <SchemaIcon />,
          path: '/reporting/data-models',
          permission: null
        }
      ]
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
      permission: 'system:configure'
    }
  ];

  // Handle drawer toggle
  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  // Handle user menu open
  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  // Handle user menu close
  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  // Handle logout
  const handleLogout = () => {
    logout();
    navigate('/login');
    handleUserMenuClose();
  };

  // Handle profile click
  const handleProfileClick = () => {
    navigate('/settings/profile');
    handleUserMenuClose();
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          width: { md: drawerOpen ? `calc(100% - ${drawerWidth}px)` : '100%' },
          ml: { md: drawerOpen ? '0' : 0 }, // Changed from ${drawerWidth}px to 0
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            {drawerOpen ? <ChevronLeftIcon /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Agentor BackOffice
          </Typography>
          <IconButton
            onClick={handleUserMenuOpen}
            color="inherit"
            edge="end"
          >
            <Avatar sx={{ bgcolor: theme.palette.secondary.main }}>
              {user?.username.charAt(0).toUpperCase()}
            </Avatar>
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleUserMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem onClick={handleProfileClick}>
              <ListItemIcon>
                <PersonIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Profile</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Logout</ListItemText>
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={drawerOpen}
        onClose={isMobile ? handleDrawerToggle : undefined}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', height: '100%' }}>
          <List>
            {menuItems.map((item) => {
              // Skip items the user doesn't have permission for
              if (item.permission && !hasPermission(item.permission)) {
                return null;
              }

              // For parent items with children (expandable menus)
              if (item.children) {
                const isActive = location.pathname.startsWith(`/reports`) ||
                                 location.pathname.startsWith(`/data-sources`) ||
                                 location.pathname.startsWith(`/reporting`);

                return (
                  <React.Fragment key={item.text}>
                    <ListItem
                      component="button"
                      onClick={() => handleMenuExpand(item.id)}
                      sx={{
                        bgcolor: isActive ? 'rgba(0, 0, 0, 0.08)' : 'transparent',
                        '&:hover': {
                          bgcolor: isActive ? 'rgba(0, 0, 0, 0.12)' : 'rgba(0, 0, 0, 0.04)',
                        },
                        textAlign: 'left',
                        width: '100%',
                        display: 'flex',
                        padding: '8px 16px',
                        border: 'none',
                        background: 'none',
                      }}
                    >
                      <ListItemIcon sx={{ color: isActive ? theme.palette.primary.main : 'inherit' }}>
                        {item.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={item.text}
                        primaryTypographyProps={{
                          color: isActive ? theme.palette.primary.main : 'inherit',
                          fontWeight: isActive ? 'bold' : 'normal',
                        }}
                      />
                      {expandedMenus[item.id] ? <ExpandLess /> : <ExpandMore />}
                    </ListItem>
                    <Collapse in={expandedMenus[item.id]} timeout="auto" unmountOnExit>
                      <List component="div" disablePadding>
                        {item.children.map((child) => {
                          if (child.permission && !hasPermission(child.permission)) {
                            return null;
                          }

                          const isChildActive = location.pathname === child.path ||
                                              location.pathname.startsWith(`${child.path}/`);

                          return (
                            <ListItem

                              key={child.text}
                              component={Link}
                              to={child.path}
                              sx={{
                                pl: 4,
                                bgcolor: isChildActive ? 'rgba(0, 0, 0, 0.08)' : 'transparent',
                                '&:hover': {
                                  bgcolor: isChildActive ? 'rgba(0, 0, 0, 0.12)' : 'rgba(0, 0, 0, 0.04)',
                                },
                              }}
                            >
                              <ListItemIcon sx={{ color: isChildActive ? theme.palette.primary.main : 'inherit' }}>
                                {child.icon}
                              </ListItemIcon>
                              <ListItemText
                                primary={child.text}
                                primaryTypographyProps={{
                                  color: isChildActive ? theme.palette.primary.main : 'inherit',
                                  fontWeight: isChildActive ? 'bold' : 'normal',
                                }}
                              />
                            </ListItem>
                          );
                        })}
                      </List>
                    </Collapse>
                  </React.Fragment>
                );
              }

              // For regular menu items without children
              const isActive = location.pathname === item.path ||
                               (item.path && location.pathname.startsWith(`${item.path}/`));

              return (
                <ListItem

                  key={item.text}
                  component={Link}
                  to={item.path}
                  sx={{
                    bgcolor: isActive ? 'rgba(0, 0, 0, 0.08)' : 'transparent',
                    '&:hover': {
                      bgcolor: isActive ? 'rgba(0, 0, 0, 0.12)' : 'rgba(0, 0, 0, 0.04)',
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: isActive ? theme.palette.primary.main : 'inherit' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.text}
                    primaryTypographyProps={{
                      color: isActive ? theme.palette.primary.main : 'inherit',
                      fontWeight: isActive ? 'bold' : 'normal',
                    }}
                  />
                </ListItem>
              );
            })}
          </List>
          <Divider />
          <Box sx={{ p: 2, mt: 'auto' }}>
            <Typography variant="body2" color="text.secondary">
              {user?.role ? `${user.role.charAt(0).toUpperCase() + user.role.slice(1)} Role` : 'User Role'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {user?.username}
            </Typography>
          </Box>
        </Box>
      </Drawer>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 1, sm: 2 },
          pl: { xs: 1, sm: 1 },
          width: { md: drawerOpen ? `calc(100% - ${drawerWidth}px)` : '100%' },
          ml: { md: drawerOpen ? '0' : 0 }, // Changed from ${drawerWidth}px to 0
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'auto'
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <Box sx={{ flexGrow: 1 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default MainLayout;
