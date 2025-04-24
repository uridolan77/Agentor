import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Tabs,
  Tab,
  Grid,
  Alert
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { PageHeader } from '../../components/common';
import { 
  UserProfileSettings, 
  SystemSettings, 
  TeamManagement,
  UserProfile,
  SystemSettingsData,
  Team,
  User
} from '../../components/settings';

// Mock user profile data
const mockUserProfile: UserProfile = {
  id: 1,
  username: 'johndoe',
  email: 'john.doe@example.com',
  firstName: 'John',
  lastName: 'Doe',
  avatar: 'https://i.pravatar.cc/150?img=1',
  role: 'Administrator',
  team: 'Engineering',
  createdAt: '2024-01-01T00:00:00Z',
  lastLogin: '2025-04-20T12:00:00Z'
};

// Mock system settings data
const mockSystemSettings: SystemSettingsData = {
  logLevel: 'info',
  maxConcurrentTasks: 5,
  enableTelemetry: true,
  defaultLLMProvider: 'openai',
  defaultAgentTimeout: 60,
  enableNotifications: true,
  notificationEmail: 'admin@example.com',
  backupFrequency: 'daily',
  theme: 'system',
  maxLogRetentionDays: 30
};

// Mock teams data
const mockTeams: Team[] = [
  {
    id: 1,
    name: 'Engineering',
    description: 'Software development team',
    createdAt: '2024-01-01T00:00:00Z',
    createdBy: 1,
    memberCount: 5
  },
  {
    id: 2,
    name: 'Operations',
    description: 'System operations team',
    createdAt: '2024-01-02T00:00:00Z',
    createdBy: 1,
    memberCount: 3
  },
  {
    id: 3,
    name: 'Research',
    description: 'AI research team',
    createdAt: '2024-01-03T00:00:00Z',
    createdBy: 1,
    memberCount: 2
  }
];

// Mock users data
const mockUsers: User[] = [
  {
    id: 1,
    username: 'johndoe',
    email: 'john.doe@example.com',
    firstName: 'John',
    lastName: 'Doe',
    avatar: 'https://i.pravatar.cc/150?img=1',
    role: 'Administrator',
    teamId: 1,
    status: 'active',
    lastLogin: '2025-04-20T12:00:00Z'
  },
  {
    id: 2,
    username: 'janedoe',
    email: 'jane.doe@example.com',
    firstName: 'Jane',
    lastName: 'Doe',
    avatar: 'https://i.pravatar.cc/150?img=2',
    role: 'Developer',
    teamId: 1,
    status: 'active',
    lastLogin: '2025-04-19T10:00:00Z'
  },
  {
    id: 3,
    username: 'bobsmith',
    email: 'bob.smith@example.com',
    firstName: 'Bob',
    lastName: 'Smith',
    avatar: 'https://i.pravatar.cc/150?img=3',
    role: 'Operator',
    teamId: 2,
    status: 'active',
    lastLogin: '2025-04-18T09:00:00Z'
  },
  {
    id: 4,
    username: 'alicejones',
    email: 'alice.jones@example.com',
    firstName: 'Alice',
    lastName: 'Jones',
    avatar: 'https://i.pravatar.cc/150?img=4',
    role: 'Researcher',
    teamId: 3,
    status: 'active',
    lastLogin: '2025-04-17T14:00:00Z'
  },
  {
    id: 5,
    username: 'charliegreen',
    email: 'charlie.green@example.com',
    firstName: 'Charlie',
    lastName: 'Green',
    role: 'Guest',
    status: 'pending'
  }
];

const SettingsPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const { user, hasPermission } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile>(mockUserProfile);
  const [systemSettings, setSystemSettings] = useState<SystemSettingsData>(mockSystemSettings);
  const [teams, setTeams] = useState<Team[]>(mockTeams);
  const [users, setUsers] = useState<User[]>(mockUsers);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle save user profile
  const handleSaveUserProfile = async (profile: UserProfile) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.put('/api/users/profile', profile);
      
      // Update state
      setUserProfile(profile);
      
      // Show success message
      console.log('Profile saved successfully');
    } catch (err: any) {
      console.error('Error saving profile:', err);
      setError('Failed to save profile. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle save system settings
  const handleSaveSystemSettings = async (settings: SystemSettingsData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.put('/api/settings/system', settings);
      
      // Update state
      setSystemSettings(settings);
      
      // Show success message
      console.log('System settings saved successfully');
    } catch (err: any) {
      console.error('Error saving system settings:', err);
      setError('Failed to save system settings. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle reset system settings
  const handleResetSystemSettings = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.post('/api/settings/system/reset');
      
      // Reset to default settings
      setSystemSettings({
        logLevel: 'info',
        maxConcurrentTasks: 5,
        enableTelemetry: true,
        defaultLLMProvider: 'openai',
        defaultAgentTimeout: 60,
        enableNotifications: false,
        backupFrequency: 'weekly',
        theme: 'system',
        maxLogRetentionDays: 30
      });
      
      // Show success message
      console.log('System settings reset successfully');
    } catch (err: any) {
      console.error('Error resetting system settings:', err);
      setError('Failed to reset system settings. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle create team
  const handleCreateTeam = async (team: Partial<Team>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.post('/api/teams', team);
      // const newTeam = response.data;
      
      // Create new team with mock ID
      const newTeam: Team = {
        id: teams.length + 1,
        name: team.name || 'New Team',
        description: team.description,
        createdAt: new Date().toISOString(),
        createdBy: 1,
        memberCount: 0
      };
      
      // Update state
      setTeams([...teams, newTeam]);
      
      // Show success message
      console.log('Team created successfully');
    } catch (err: any) {
      console.error('Error creating team:', err);
      setError('Failed to create team. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle update team
  const handleUpdateTeam = async (id: number, team: Partial<Team>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.put(`/api/teams/${id}`, team);
      
      // Update state
      setTeams(teams.map(t => t.id === id ? { ...t, ...team } : t));
      
      // Show success message
      console.log('Team updated successfully');
    } catch (err: any) {
      console.error('Error updating team:', err);
      setError('Failed to update team. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle delete team
  const handleDeleteTeam = async (id: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/teams/${id}`);
      
      // Update state
      setTeams(teams.filter(t => t.id !== id));
      
      // Show success message
      console.log('Team deleted successfully');
    } catch (err: any) {
      console.error('Error deleting team:', err);
      setError('Failed to delete team. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle invite user
  const handleInviteUser = async (email: string, teamId?: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.post('/api/users/invite', { email, teamId });
      // const newUser = response.data;
      
      // Create new user with mock ID
      const newUser: User = {
        id: users.length + 1,
        username: email.split('@')[0],
        email,
        firstName: '',
        lastName: '',
        role: 'User',
        teamId,
        status: 'pending'
      };
      
      // Update state
      setUsers([...users, newUser]);
      
      // Show success message
      console.log('User invited successfully');
    } catch (err: any) {
      console.error('Error inviting user:', err);
      setError('Failed to invite user. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle update user
  const handleUpdateUser = async (id: number, user: Partial<User>) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.put(`/api/users/${id}`, user);
      
      // Update state
      setUsers(users.map(u => u.id === id ? { ...u, ...user } : u));
      
      // Show success message
      console.log('User updated successfully');
    } catch (err: any) {
      console.error('Error updating user:', err);
      setError('Failed to update user. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Handle delete user
  const handleDeleteUser = async (id: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call
      // await axios.delete(`/api/users/${id}`);
      
      // Update state
      setUsers(users.filter(u => u.id !== id));
      
      // Show success message
      console.log('User deleted successfully');
    } catch (err: any) {
      console.error('Error deleting user:', err);
      setError('Failed to delete user. Please try again.');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box>
      <PageHeader 
        title="Settings"
        breadcrumbs={[{ label: 'Settings' }]}
      />
      
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Profile" />
          {hasPermission('user:read') && <Tab label="Users & Teams" />}
          {hasPermission('system:configure') && <Tab label="System" />}
        </Tabs>
      </Paper>
      
      {/* Profile Tab */}
      {tabValue === 0 && (
        <UserProfileSettings
          profile={userProfile}
          onSave={handleSaveUserProfile}
          isLoading={isLoading}
          error={error}
        />
      )}
      
      {/* Users & Teams Tab */}
      {tabValue === 1 && hasPermission('user:read') && (
        <TeamManagement
          teams={teams}
          users={users}
          isLoading={isLoading}
          error={error}
          onCreateTeam={handleCreateTeam}
          onUpdateTeam={handleUpdateTeam}
          onDeleteTeam={handleDeleteTeam}
          onInviteUser={handleInviteUser}
          onUpdateUser={handleUpdateUser}
          onDeleteUser={handleDeleteUser}
        />
      )}
      
      {/* System Tab */}
      {tabValue === 2 && hasPermission('system:configure') && (
        <SystemSettings
          settings={systemSettings}
          onSave={handleSaveSystemSettings}
          onReset={handleResetSystemSettings}
          isLoading={isLoading}
          error={error}
        />
      )}
    </Box>
  );
};

export default SettingsPage;
