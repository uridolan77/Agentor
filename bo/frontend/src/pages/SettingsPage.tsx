import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Alert,
  Snackbar,
  Grid
} from '@mui/material';
import {
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
  PersonOutline as PersonIcon,
  Language as LanguageIcon
} from '@mui/icons-material';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
}

const SettingsPage = () => {
  const [tabValue, setTabValue] = useState(0);
  const [darkMode, setDarkMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [language, setLanguage] = useState('english');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSaveAppearance = () => {
    // Save appearance settings logic
    setNotification({
      open: true,
      message: 'Appearance settings saved successfully',
      severity: 'success'
    });
  };

  const handleSaveNotifications = () => {
    // Save notification settings logic
    setNotification({
      open: true,
      message: 'Notification settings saved successfully',
      severity: 'success'
    });
  };

  const handleSaveLanguage = () => {
    // Save language settings logic
    setNotification({
      open: true,
      message: 'Language settings saved successfully',
      severity: 'success'
    });
  };

  const handleChangePassword = () => {
    // Password validation
    if (newPassword !== confirmPassword) {
      setNotification({
        open: true,
        message: 'New passwords do not match',
        severity: 'error'
      });
      return;
    }
    
    if (newPassword.length < 8) {
      setNotification({
        open: true,
        message: 'Password must be at least 8 characters',
        severity: 'error'
      });
      return;
    }
    
    // Change password logic
    setNotification({
      open: true,
      message: 'Password updated successfully',
      severity: 'success'
    });
    
    // Clear password fields
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="settings tabs">
            <Tab label="Appearance" icon={<DarkModeIcon />} iconPosition="start" {...a11yProps(0)} />
            <Tab label="Notifications" icon={<NotificationsIcon />} iconPosition="start" {...a11yProps(1)} />
            <Tab label="Security" icon={<SecurityIcon />} iconPosition="start" {...a11yProps(2)} />
            <Tab label="Language" icon={<LanguageIcon />} iconPosition="start" {...a11yProps(3)} />
          </Tabs>
        </Box>
        
        {/* Appearance Settings */}
        <TabPanel value={tabValue} index={0}>
          <Typography variant="h6" gutterBottom>
            Theme Settings
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                {darkMode ? <DarkModeIcon /> : <LightModeIcon />}
              </ListItemIcon>
              <ListItemText 
                primary="Dark Mode" 
                secondary="Switch between light and dark theme" 
              />
              <Switch
                edge="end"
                checked={darkMode}
                onChange={() => setDarkMode(!darkMode)}
                inputProps={{
                  'aria-labelledby': 'dark-mode-switch',
                }}
              />
            </ListItem>
            
            <Divider />
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button variant="contained" color="primary" onClick={handleSaveAppearance}>
                Save Changes
              </Button>
            </Box>
          </List>
        </TabPanel>
        
        {/* Notifications Settings */}
        <TabPanel value={tabValue} index={1}>
          <Typography variant="h6" gutterBottom>
            Notification Preferences
          </Typography>
          
          <List>
            <ListItem>
              <ListItemIcon>
                <NotificationsIcon />
              </ListItemIcon>
              <ListItemText 
                primary="Enable Notifications" 
                secondary="Receive in-app notifications" 
              />
              <Switch
                edge="end"
                checked={notificationsEnabled}
                onChange={() => setNotificationsEnabled(!notificationsEnabled)}
              />
            </ListItem>
            
            <Divider />
            
            <ListItem>
              <ListItemIcon>
                <NotificationsIcon />
              </ListItemIcon>
              <ListItemText 
                primary="Email Notifications" 
                secondary="Receive email notifications" 
              />
              <Switch
                edge="end"
                checked={emailNotifications}
                onChange={() => setEmailNotifications(!emailNotifications)}
                disabled={!notificationsEnabled}
              />
            </ListItem>
            
            <Divider />
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleSaveNotifications}
              >
                Save Changes
              </Button>
            </Box>
          </List>
        </TabPanel>
        
        {/* Security Settings */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Change Password
          </Typography>
          
          <Box component="form" noValidate autoComplete="off" sx={{ mt: 2 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="Current Password"
                  variant="outlined"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  required
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="New Password"
                  variant="outlined"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                  helperText="Password must be at least 8 characters long"
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="Confirm New Password"
                  variant="outlined"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  error={newPassword !== confirmPassword && confirmPassword !== ''}
                  helperText={
                    newPassword !== confirmPassword && confirmPassword !== '' 
                      ? "Passwords don't match" 
                      : ''
                  }
                />
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleChangePassword}
                disabled={!currentPassword || !newPassword || !confirmPassword}
              >
                Update Password
              </Button>
            </Box>
          </Box>
        </TabPanel>
        
        {/* Language Settings */}
        <TabPanel value={tabValue} index={3}>
          <Typography variant="h6" gutterBottom>
            Language Settings
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <TextField
              select
              fullWidth
              label="Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              SelectProps={{
                native: true,
              }}
              variant="outlined"
            >
              <option value="english">English</option>
              <option value="spanish">Spanish</option>
              <option value="french">French</option>
              <option value="german">German</option>
              <option value="chinese">Chinese</option>
              <option value="japanese">Japanese</option>
            </TextField>
            
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleSaveLanguage}
              >
                Save Changes
              </Button>
            </Box>
          </Box>
        </TabPanel>
      </Paper>
      
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

export default SettingsPage;