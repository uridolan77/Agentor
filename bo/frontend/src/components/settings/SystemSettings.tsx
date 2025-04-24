import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Switch,
  Slider,
  TextField,
  MenuItem,
  FormControlLabel
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { FormField } from '../common';

export interface SystemSettingsData {
  logLevel: 'debug' | 'info' | 'warning' | 'error';
  maxConcurrentTasks: number;
  enableTelemetry: boolean;
  defaultLLMProvider: string;
  defaultAgentTimeout: number;
  enableNotifications: boolean;
  notificationEmail?: string;
  backupFrequency: 'daily' | 'weekly' | 'monthly' | 'never';
  theme: 'light' | 'dark' | 'system';
  maxLogRetentionDays: number;
}

interface SystemSettingsProps {
  settings: SystemSettingsData;
  onSave: (settings: SystemSettingsData) => Promise<void>;
  onReset: () => Promise<void>;
  isLoading?: boolean;
  error?: string | null;
}

/**
 * SystemSettings component for configuring system-wide settings.
 */
const SystemSettings: React.FC<SystemSettingsProps> = ({
  settings,
  onSave,
  onReset,
  isLoading = false,
  error = null
}) => {
  const [formValues, setFormValues] = useState<SystemSettingsData>(settings);

  // Handle field change
  const handleChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      await onSave(formValues);
    } catch (err) {
      console.error('Error saving settings:', err);
    }
  };

  // Handle reset to defaults
  const handleReset = async () => {
    try {
      await onReset();
    } catch (err) {
      console.error('Error resetting settings:', err);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">System Settings</Typography>
        <Button
          variant="outlined"
          color="secondary"
          startIcon={<RefreshIcon />}
          onClick={handleReset}
          disabled={isLoading}
        >
          Reset to Defaults
        </Button>
      </Box>
      
      <Divider sx={{ mb: 3 }} />
      
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {/* General Settings */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>
              General Settings
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="theme"
              label="Theme"
              value={formValues.theme}
              onChange={handleChange}
              options={[
                { value: 'light', label: 'Light' },
                { value: 'dark', label: 'Dark' },
                { value: 'system', label: 'System Default' }
              ]}
              disabled={isLoading}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="logLevel"
              label="Log Level"
              value={formValues.logLevel}
              onChange={handleChange}
              options={[
                { value: 'debug', label: 'Debug' },
                { value: 'info', label: 'Info' },
                { value: 'warning', label: 'Warning' },
                { value: 'error', label: 'Error' }
              ]}
              disabled={isLoading}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="switch"
              name="enableTelemetry"
              label="Enable Telemetry"
              value={formValues.enableTelemetry}
              onChange={handleChange}
              disabled={isLoading}
              helperText="Send anonymous usage data to help improve the platform"
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="number"
              name="maxLogRetentionDays"
              label="Log Retention (days)"
              value={formValues.maxLogRetentionDays}
              onChange={handleChange}
              min={1}
              max={365}
              disabled={isLoading}
            />
          </Grid>
          
          {/* Performance Settings */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Performance Settings
            </Typography>
            <Divider />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="number"
              name="maxConcurrentTasks"
              label="Max Concurrent Tasks"
              value={formValues.maxConcurrentTasks}
              onChange={handleChange}
              min={1}
              max={50}
              disabled={isLoading}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="number"
              name="defaultAgentTimeout"
              label="Default Agent Timeout (seconds)"
              value={formValues.defaultAgentTimeout}
              onChange={handleChange}
              min={10}
              max={3600}
              disabled={isLoading}
            />
          </Grid>
          
          {/* Integration Settings */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Integration Settings
            </Typography>
            <Divider />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="defaultLLMProvider"
              label="Default LLM Provider"
              value={formValues.defaultLLMProvider}
              onChange={handleChange}
              options={[
                { value: 'openai', label: 'OpenAI' },
                { value: 'anthropic', label: 'Anthropic' },
                { value: 'google', label: 'Google AI' },
                { value: 'local', label: 'Local Model' }
              ]}
              disabled={isLoading}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="select"
              name="backupFrequency"
              label="Backup Frequency"
              value={formValues.backupFrequency}
              onChange={handleChange}
              options={[
                { value: 'daily', label: 'Daily' },
                { value: 'weekly', label: 'Weekly' },
                { value: 'monthly', label: 'Monthly' },
                { value: 'never', label: 'Never' }
              ]}
              disabled={isLoading}
            />
          </Grid>
          
          {/* Notification Settings */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Notification Settings
            </Typography>
            <Divider />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormField
              type="switch"
              name="enableNotifications"
              label="Enable Email Notifications"
              value={formValues.enableNotifications}
              onChange={handleChange}
              disabled={isLoading}
            />
          </Grid>
          
          {formValues.enableNotifications && (
            <Grid item xs={12} md={6}>
              <FormField
                type="email"
                name="notificationEmail"
                label="Notification Email"
                value={formValues.notificationEmail || ''}
                onChange={handleChange}
                disabled={isLoading}
                required={formValues.enableNotifications}
              />
            </Grid>
          )}
          
          {/* Form Actions */}
          <Grid item xs={12} sx={{ mt: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                startIcon={isLoading ? <CircularProgress size={20} /> : <SaveIcon />}
                disabled={isLoading}
              >
                Save Settings
              </Button>
            </Box>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default SystemSettings;
