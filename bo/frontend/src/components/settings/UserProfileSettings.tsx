import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Button,
  Grid,
  Avatar,
  IconButton,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Edit as EditIcon,
  PhotoCamera as PhotoCameraIcon,
  Save as SaveIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { FormField } from '../common';

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  role: string;
  team?: string;
  createdAt: string;
  lastLogin?: string;
}

interface UserProfileSettingsProps {
  profile: UserProfile;
  onSave: (profile: UserProfile) => Promise<void>;
  isLoading?: boolean;
  error?: string | null;
}

/**
 * UserProfileSettings component for displaying and editing user profile settings.
 */
const UserProfileSettings: React.FC<UserProfileSettingsProps> = ({
  profile,
  onSave,
  isLoading = false,
  error = null
}) => {
  const [editMode, setEditMode] = useState(false);
  const [formValues, setFormValues] = useState<Partial<UserProfile>>(profile);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreview, setAvatarPreview] = useState<string | undefined>(profile.avatar);

  // Handle edit mode toggle
  const handleEditToggle = () => {
    if (editMode) {
      // Cancel edit mode
      setFormValues(profile);
      setAvatarPreview(profile.avatar);
      setAvatarFile(null);
    }
    setEditMode(!editMode);
  };

  // Handle field change
  const handleChange = (name: string, value: any) => {
    setFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle avatar change
  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAvatarFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setAvatarPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      // In a real app, you would upload the avatar file here
      // and get back a URL to include in the profile update
      
      await onSave({
        ...profile,
        ...formValues,
        avatar: avatarPreview
      });
      
      setEditMode(false);
    } catch (err) {
      console.error('Error saving profile:', err);
    }
  };

  // Format date
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <Paper sx={{ p: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">User Profile</Typography>
        <Button
          variant={editMode ? 'outlined' : 'contained'}
          color={editMode ? 'error' : 'primary'}
          startIcon={editMode ? <CancelIcon /> : <EditIcon />}
          onClick={handleEditToggle}
          disabled={isLoading}
        >
          {editMode ? 'Cancel' : 'Edit Profile'}
        </Button>
      </Box>
      
      <Divider sx={{ mb: 3 }} />
      
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {/* Avatar */}
          <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
            <Box sx={{ position: 'relative', display: 'inline-block' }}>
              <Avatar
                src={avatarPreview}
                alt={`${formValues.firstName} ${formValues.lastName}`}
                sx={{ width: 150, height: 150, mb: 2 }}
              />
              {editMode && (
                <Box sx={{ position: 'absolute', bottom: 16, right: 0 }}>
                  <input
                    accept="image/*"
                    style={{ display: 'none' }}
                    id="avatar-upload"
                    type="file"
                    onChange={handleAvatarChange}
                  />
                  <label htmlFor="avatar-upload">
                    <IconButton
                      color="primary"
                      component="span"
                      sx={{ bgcolor: 'background.paper' }}
                    >
                      <PhotoCameraIcon />
                    </IconButton>
                  </label>
                </Box>
              )}
            </Box>
            
            <Typography variant="subtitle1" gutterBottom>
              {profile.firstName} {profile.lastName}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {profile.role}
            </Typography>
            {profile.team && (
              <Typography variant="body2" color="text.secondary">
                Team: {profile.team}
              </Typography>
            )}
          </Grid>
          
          {/* Profile Fields */}
          <Grid item xs={12} md={8}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormField
                  type="text"
                  name="firstName"
                  label="First Name"
                  value={formValues.firstName || ''}
                  onChange={handleChange}
                  disabled={!editMode || isLoading}
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormField
                  type="text"
                  name="lastName"
                  label="Last Name"
                  value={formValues.lastName || ''}
                  onChange={handleChange}
                  disabled={!editMode || isLoading}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <FormField
                  type="email"
                  name="email"
                  label="Email"
                  value={formValues.email || ''}
                  onChange={handleChange}
                  disabled={!editMode || isLoading}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <FormField
                  type="text"
                  name="username"
                  label="Username"
                  value={formValues.username || ''}
                  onChange={handleChange}
                  disabled={true} // Username cannot be changed
                  helperText="Username cannot be changed"
                />
              </Grid>
              
              {!editMode && (
                <>
                  <Grid item xs={12} md={6}>
                    <Typography variant="caption" color="text.secondary">
                      Account Created
                    </Typography>
                    <Typography variant="body2">
                      {formatDate(profile.createdAt)}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="caption" color="text.secondary">
                      Last Login
                    </Typography>
                    <Typography variant="body2">
                      {formatDate(profile.lastLogin)}
                    </Typography>
                  </Grid>
                </>
              )}
            </Grid>
            
            {editMode && (
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  startIcon={isLoading ? <CircularProgress size={20} /> : <SaveIcon />}
                  disabled={isLoading}
                >
                  Save Changes
                </Button>
              </Box>
            )}
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default UserProfileSettings;
