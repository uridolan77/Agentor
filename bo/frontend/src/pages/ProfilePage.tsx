import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Avatar,
  Button,
  TextField,
  Grid,
  Divider,
  IconButton,
  Skeleton,
  Alert,
  Snackbar,
  Card,
  CardContent
} from '@mui/material';
import {
  Edit as EditIcon,
  PhotoCamera as PhotoCameraIcon,
  Save as SaveIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';

interface Profile {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  jobTitle: string;
  department: string;
  phone: string;
  avatar: string;
  bio: string;
}

const ProfilePage = () => {
  const [profile, setProfile] = useState<Profile>({
    id: 1,
    firstName: 'John',
    lastName: 'Doe',
    email: 'john.doe@example.com',
    jobTitle: 'Software Engineer',
    department: 'Engineering',
    phone: '+1 (555) 123-4567',
    avatar: '',
    bio: 'Experienced software engineer with a focus on AI and machine learning applications.'
  });
  
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editedProfile, setEditedProfile] = useState<Profile | null>(null);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Fetch profile data
  useEffect(() => {
    // Simulate API call
    setLoading(true);
    setTimeout(() => {
      // In a real application, you would fetch the profile data from your API
      setLoading(false);
    }, 800);
  }, []);

  const handleEdit = () => {
    setEditedProfile({ ...profile });
    setEditing(true);
  };

  const handleCancelEdit = () => {
    setEditedProfile(null);
    setEditing(false);
  };

  const handleSaveProfile = () => {
    if (!editedProfile) return;

    // Simulate API call to save profile changes
    setLoading(true);
    
    setTimeout(() => {
      setProfile(editedProfile);
      setEditing(false);
      setEditedProfile(null);
      setLoading(false);
      
      setNotification({
        open: true,
        message: 'Profile updated successfully!',
        severity: 'success'
      });
    }, 800);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    if (editedProfile) {
      setEditedProfile({
        ...editedProfile,
        [name]: value
      });
    }
  };

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      // In a real application, you would upload the file to your server
      // For this example, we'll just use a local URL
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        if (editedProfile && event.target?.result) {
          setEditedProfile({
            ...editedProfile,
            avatar: event.target.result as string
          });
        }
      };
      fileReader.readAsDataURL(e.target.files[0]);
    }
  };

  // Format full name for display
  const fullName = `${profile.firstName} ${profile.lastName}`;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        My Profile
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          {/* Profile Header */}
          <Grid item xs={12} md={3} sx={{ textAlign: 'center', mb: { xs: 2, md: 0 } }}>
            {loading ? (
              <Skeleton 
                variant="circular" 
                width={150} 
                height={150} 
                sx={{ margin: '0 auto' }} 
              />
            ) : (
              <Box position="relative" display="inline-block">
                <Avatar
                  src={editing ? editedProfile?.avatar : profile.avatar}
                  alt={fullName}
                  sx={{
                    width: 150,
                    height: 150,
                    border: '3px solid',
                    borderColor: 'primary.main',
                    margin: '0 auto',
                  }}
                >
                  {fullName.charAt(0)}
                </Avatar>
                
                {editing && (
                  <Box 
                    position="absolute" 
                    bottom="8px" 
                    right="8px" 
                    zIndex={1}
                    sx={{
                      bgcolor: 'primary.main',
                      borderRadius: '50%',
                    }}
                  >
                    <IconButton 
                      color="primary" 
                      aria-label="upload picture" 
                      component="label"
                      sx={{ 
                        color: 'white',
                        p: 0.5
                      }}
                    >
                      <input 
                        hidden 
                        accept="image/*" 
                        type="file" 
                        onChange={handleAvatarChange} 
                      />
                      <PhotoCameraIcon />
                    </IconButton>
                  </Box>
                )}
              </Box>
            )}
            
            <Typography variant="h5" sx={{ mt: 2 }}>
              {loading ? <Skeleton width="100%" /> : fullName}
            </Typography>
            
            <Typography variant="body1" color="text.secondary">
              {loading ? <Skeleton width="80%" /> : profile.jobTitle}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {loading ? <Skeleton width="60%" /> : profile.department}
            </Typography>
            
            {!editing && !loading && (
              <Button
                variant="outlined"
                startIcon={<EditIcon />}
                sx={{ mt: 2 }}
                onClick={handleEdit}
              >
                Edit Profile
              </Button>
            )}
            
            {editing && (
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 1 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSaveProfile}
                >
                  Save
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<CancelIcon />}
                  onClick={handleCancelEdit}
                >
                  Cancel
                </Button>
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12} md={9}>
            {/* Profile Info */}
            <Box>
              <Typography variant="h6" gutterBottom>
                Personal Information
              </Typography>
              
              {loading ? (
                <Box sx={{ mb: 3 }}>
                  <Skeleton height={60} />
                  <Skeleton height={60} />
                  <Skeleton height={60} />
                </Box>
              ) : editing ? (
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="First Name"
                      name="firstName"
                      value={editedProfile?.firstName || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Last Name"
                      name="lastName"
                      value={editedProfile?.lastName || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      name="email"
                      type="email"
                      value={editedProfile?.email || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Job Title"
                      name="jobTitle"
                      value={editedProfile?.jobTitle || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Department"
                      name="department"
                      value={editedProfile?.department || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Phone"
                      name="phone"
                      value={editedProfile?.phone || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Bio"
                      name="bio"
                      multiline
                      rows={4}
                      value={editedProfile?.bio || ''}
                      onChange={handleChange}
                    />
                  </Grid>
                </Grid>
              ) : (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Email
                    </Typography>
                    <Typography variant="body1">{profile.email}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Phone
                    </Typography>
                    <Typography variant="body1">{profile.phone}</Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" color="text.secondary">
                      Bio
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {profile.bio}
                    </Typography>
                  </Grid>
                </Grid>
              )}
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Recent Activity */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Recent Activity
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1">
                  Created report: Monthly Sales Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  2 days ago
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1">
                  Updated data source: Customer Database
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  5 days ago
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
      
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

export default ProfilePage;