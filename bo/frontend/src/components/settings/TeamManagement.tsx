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
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Chip,
  Avatar
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PersonAdd as InviteIcon,
  Mail as MailIcon
} from '@mui/icons-material';
import { DataTable, Column, ConfirmDialog, FormField } from '../common';

export interface User {
  id: number;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  role: string;
  teamId?: number;
  status: 'active' | 'inactive' | 'pending';
  lastLogin?: string;
}

export interface Team {
  id: number;
  name: string;
  description?: string;
  createdAt: string;
  createdBy: number;
  memberCount: number;
}

interface TeamManagementProps {
  teams: Team[];
  users: User[];
  isLoading?: boolean;
  error?: string | null;
  onCreateTeam: (team: Partial<Team>) => Promise<void>;
  onUpdateTeam: (id: number, team: Partial<Team>) => Promise<void>;
  onDeleteTeam: (id: number) => Promise<void>;
  onInviteUser: (email: string, teamId?: number) => Promise<void>;
  onUpdateUser: (id: number, user: Partial<User>) => Promise<void>;
  onDeleteUser: (id: number) => Promise<void>;
}

/**
 * TeamManagement component for managing teams and users.
 */
const TeamManagement: React.FC<TeamManagementProps> = ({
  teams,
  users,
  isLoading = false,
  error = null,
  onCreateTeam,
  onUpdateTeam,
  onDeleteTeam,
  onInviteUser,
  onUpdateUser,
  onDeleteUser
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [teamDialogOpen, setTeamDialogOpen] = useState(false);
  const [userDialogOpen, setUserDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [teamFormValues, setTeamFormValues] = useState<Partial<Team>>({
    name: '',
    description: ''
  });
  const [userFormValues, setUserFormValues] = useState<Partial<User>>({
    email: '',
    teamId: undefined
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle team dialog open
  const handleTeamDialogOpen = (team?: Team) => {
    if (team) {
      setSelectedTeam(team);
      setTeamFormValues({
        name: team.name,
        description: team.description
      });
    } else {
      setSelectedTeam(null);
      setTeamFormValues({
        name: '',
        description: ''
      });
    }
    setTeamDialogOpen(true);
  };

  // Handle team dialog close
  const handleTeamDialogClose = () => {
    setTeamDialogOpen(false);
    setSelectedTeam(null);
  };

  // Handle user dialog open
  const handleUserDialogOpen = (user?: User) => {
    if (user) {
      setSelectedUser(user);
      setUserFormValues({
        email: user.email,
        teamId: user.teamId
      });
    } else {
      setSelectedUser(null);
      setUserFormValues({
        email: '',
        teamId: undefined
      });
    }
    setUserDialogOpen(true);
  };

  // Handle user dialog close
  const handleUserDialogClose = () => {
    setUserDialogOpen(false);
    setSelectedUser(null);
  };

  // Handle delete dialog open
  const handleDeleteDialogOpen = (type: 'team' | 'user', id: number) => {
    if (type === 'team') {
      const team = teams.find(t => t.id === id);
      if (team) {
        setSelectedTeam(team);
        setSelectedUser(null);
      }
    } else {
      const user = users.find(u => u.id === id);
      if (user) {
        setSelectedUser(user);
        setSelectedTeam(null);
      }
    }
    setDeleteDialogOpen(true);
  };

  // Handle delete dialog close
  const handleDeleteDialogClose = () => {
    setDeleteDialogOpen(false);
    setSelectedTeam(null);
    setSelectedUser(null);
  };

  // Handle team form change
  const handleTeamFormChange = (name: string, value: any) => {
    setTeamFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle user form change
  const handleUserFormChange = (name: string, value: any) => {
    setUserFormValues(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Handle team form submit
  const handleTeamFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!teamFormValues.name) return;
    
    setIsSubmitting(true);
    
    try {
      if (selectedTeam) {
        await onUpdateTeam(selectedTeam.id, teamFormValues);
      } else {
        await onCreateTeam(teamFormValues);
      }
      handleTeamDialogClose();
    } catch (err) {
      console.error('Error saving team:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle user form submit
  const handleUserFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!userFormValues.email) return;
    
    setIsSubmitting(true);
    
    try {
      if (selectedUser) {
        await onUpdateUser(selectedUser.id, userFormValues);
      } else {
        await onInviteUser(userFormValues.email!, userFormValues.teamId);
      }
      handleUserDialogClose();
    } catch (err) {
      console.error('Error saving user:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle delete confirm
  const handleDeleteConfirm = async () => {
    setIsSubmitting(true);
    
    try {
      if (selectedTeam) {
        await onDeleteTeam(selectedTeam.id);
      } else if (selectedUser) {
        await onDeleteUser(selectedUser.id);
      }
      handleDeleteDialogClose();
    } catch (err) {
      console.error('Error deleting:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Format date
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get team name by ID
  const getTeamName = (teamId?: number) => {
    if (!teamId) return 'No Team';
    const team = teams.find(t => t.id === teamId);
    return team ? team.name : 'Unknown Team';
  };

  // Team columns
  const teamColumns: Column<Team>[] = [
    {
      id: 'name',
      label: 'Team Name',
      minWidth: 150,
      sortable: true
    },
    {
      id: 'description',
      label: 'Description',
      minWidth: 200,
      sortable: false,
      format: (value) => value || '-'
    },
    {
      id: 'memberCount',
      label: 'Members',
      minWidth: 100,
      align: 'center',
      sortable: true
    },
    {
      id: 'createdAt',
      label: 'Created',
      minWidth: 150,
      sortable: true,
      format: (value) => formatDate(value)
    },
    {
      id: 'actions',
      label: 'Actions',
      minWidth: 120,
      align: 'center',
      sortable: false,
      format: (_, team) => (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Tooltip title="Edit">
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                handleTeamDialogOpen(team);
              }}
            >
              <EditIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton 
              size="small" 
              color="error"
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteDialogOpen('team', team.id);
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )
    }
  ];

  // User columns
  const userColumns: Column<User>[] = [
    {
      id: 'user',
      label: 'User',
      minWidth: 200,
      sortable: true,
      format: (_, user) => (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Avatar 
            src={user.avatar} 
            alt={`${user.firstName} ${user.lastName}`}
            sx={{ width: 32, height: 32, mr: 1 }}
          />
          <Box>
            <Typography variant="body2">
              {user.firstName} {user.lastName}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {user.username}
            </Typography>
          </Box>
        </Box>
      )
    },
    {
      id: 'email',
      label: 'Email',
      minWidth: 200,
      sortable: true
    },
    {
      id: 'role',
      label: 'Role',
      minWidth: 120,
      sortable: true
    },
    {
      id: 'teamId',
      label: 'Team',
      minWidth: 150,
      sortable: true,
      format: (value) => getTeamName(value)
    },
    {
      id: 'status',
      label: 'Status',
      minWidth: 100,
      align: 'center',
      sortable: true,
      format: (value) => (
        <Chip 
          label={value.toUpperCase()} 
          size="small"
          color={
            value === 'active' ? 'success' : 
            value === 'pending' ? 'warning' : 
            'error'
          }
        />
      )
    },
    {
      id: 'actions',
      label: 'Actions',
      minWidth: 120,
      align: 'center',
      sortable: false,
      format: (_, user) => (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Tooltip title="Edit">
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                handleUserDialogOpen(user);
              }}
            >
              <EditIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton 
              size="small" 
              color="error"
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteDialogOpen('user', user.id);
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )
    }
  ];

  return (
    <Paper sx={{ p: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Teams" />
          <Tab label="Users" />
        </Tabs>
      </Box>
      
      {/* Teams Tab */}
      {tabValue === 0 && (
        <>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={() => handleTeamDialogOpen()}
            >
              Create Team
            </Button>
          </Box>
          
          <DataTable
            columns={teamColumns}
            data={teams}
            keyExtractor={(team) => team.id}
            isLoading={isLoading}
            error={error}
            emptyMessage="No teams found."
          />
        </>
      )}
      
      {/* Users Tab */}
      {tabValue === 1 && (
        <>
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<InviteIcon />}
              onClick={() => handleUserDialogOpen()}
            >
              Invite User
            </Button>
          </Box>
          
          <DataTable
            columns={userColumns}
            data={users}
            keyExtractor={(user) => user.id}
            isLoading={isLoading}
            error={error}
            emptyMessage="No users found."
          />
        </>
      )}
      
      {/* Team Dialog */}
      <ConfirmDialog
        open={teamDialogOpen}
        title={selectedTeam ? 'Edit Team' : 'Create Team'}
        confirmLabel={selectedTeam ? 'Save' : 'Create'}
        cancelLabel="Cancel"
        onConfirm={() => handleTeamFormSubmit(new Event('submit') as unknown as React.FormEvent)}
        onCancel={handleTeamDialogClose}
        isLoading={isSubmitting}
        maxWidth="sm"
        fullWidth
      >
        <form onSubmit={handleTeamFormSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormField
                type="text"
                name="name"
                label="Team Name"
                value={teamFormValues.name || ''}
                onChange={handleTeamFormChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="textarea"
                name="description"
                label="Description"
                value={teamFormValues.description || ''}
                onChange={handleTeamFormChange}
                rows={3}
              />
            </Grid>
          </Grid>
        </form>
      </ConfirmDialog>
      
      {/* User Dialog */}
      <ConfirmDialog
        open={userDialogOpen}
        title={selectedUser ? 'Edit User' : 'Invite User'}
        confirmLabel={selectedUser ? 'Save' : 'Invite'}
        cancelLabel="Cancel"
        onConfirm={() => handleUserFormSubmit(new Event('submit') as unknown as React.FormEvent)}
        onCancel={handleUserDialogClose}
        isLoading={isSubmitting}
        maxWidth="sm"
        fullWidth
      >
        <form onSubmit={handleUserFormSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormField
                type="email"
                name="email"
                label="Email"
                value={userFormValues.email || ''}
                onChange={handleUserFormChange}
                required
                disabled={!!selectedUser}
              />
            </Grid>
            <Grid item xs={12}>
              <FormField
                type="select"
                name="teamId"
                label="Team"
                value={userFormValues.teamId || ''}
                onChange={handleUserFormChange}
                options={[
                  { value: '', label: 'No Team' },
                  ...teams.map(team => ({
                    value: team.id,
                    label: team.name
                  }))
                ]}
              />
            </Grid>
          </Grid>
        </form>
      </ConfirmDialog>
      
      {/* Delete Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        title={selectedTeam ? 'Delete Team' : 'Delete User'}
        message={
          selectedTeam
            ? `Are you sure you want to delete the team "${selectedTeam.name}"? This action cannot be undone.`
            : selectedUser
            ? `Are you sure you want to delete the user "${selectedUser.firstName} ${selectedUser.lastName}"? This action cannot be undone.`
            : 'Are you sure you want to delete this item? This action cannot be undone.'
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteDialogClose}
        isLoading={isSubmitting}
        confirmColor="error"
      />
    </Paper>
  );
};

export default TeamManagement;
