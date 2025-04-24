import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon
} from '@mui/icons-material';
import {
  DataTable,
  Column,
  SearchBar,
  StatusBadge,
  ConfirmDialog,
  EmptyState
} from '../common';

// Define Agent type based on the backend schema
export interface Agent {
  id: number;
  name: string;
  description: string | null;
  agent_type: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  creator_id: number;
  team_id: number | null;
  configuration: Record<string, any> | string;
}

interface AgentListProps {
  agents: Agent[];
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
  onDelete: (id: number) => Promise<void>;
}

/**
 * AgentList component for displaying a list of agents.
 * Includes search, filtering, and actions (view, edit, delete).
 */
const AgentList: React.FC<AgentListProps> = ({
  agents,
  isLoading,
  error,
  onRefresh,
  onDelete
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredAgents, setFilteredAgents] = useState<Agent[]>(agents);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [agentToDelete, setAgentToDelete] = useState<Agent | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Filter agents based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredAgents(agents);
      return;
    }

    const query = searchQuery.toLowerCase();
    const filtered = agents.filter(agent => 
      agent.name.toLowerCase().includes(query) || 
      (agent.description && agent.description.toLowerCase().includes(query)) ||
      agent.agent_type.toLowerCase().includes(query)
    );
    
    setFilteredAgents(filtered);
  }, [agents, searchQuery]);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  // Handle view agent
  const handleViewAgent = useCallback((agent: Agent) => {
    navigate(`/agents/${agent.id}`);
  }, [navigate]);

  // Handle edit agent
  const handleEditAgent = useCallback((agent: Agent) => {
    navigate(`/agents/${agent.id}/edit`);
  }, [navigate]);

  // Handle delete agent
  const handleDeleteClick = useCallback((agent: Agent) => {
    setAgentToDelete(agent);
    setDeleteDialogOpen(true);
  }, []);

  // Confirm delete agent
  const handleConfirmDelete = useCallback(async () => {
    if (!agentToDelete) return;
    
    setIsDeleting(true);
    try {
      await onDelete(agentToDelete.id);
      setDeleteDialogOpen(false);
      setAgentToDelete(null);
    } catch (error) {
      console.error('Error deleting agent:', error);
    } finally {
      setIsDeleting(false);
    }
  }, [agentToDelete, onDelete]);

  // Cancel delete agent
  const handleCancelDelete = useCallback(() => {
    setDeleteDialogOpen(false);
    setAgentToDelete(null);
  }, []);

  // Define table columns
  const columns: Column<Agent>[] = [
    {
      id: 'name',
      label: 'Name',
      minWidth: 150,
      sortable: true
    },
    {
      id: 'agent_type',
      label: 'Type',
      minWidth: 120,
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
      id: 'is_active',
      label: 'Status',
      minWidth: 100,
      align: 'center',
      sortable: true,
      format: (value) => (
        <StatusBadge status={value ? 'active' : 'inactive'} />
      )
    },
    {
      id: 'actions',
      label: 'Actions',
      minWidth: 120,
      align: 'center',
      sortable: false,
      format: (_, agent) => (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Tooltip title="View">
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                handleViewAgent(agent);
              }}
            >
              <ViewIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Edit">
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                handleEditAgent(agent);
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
                handleDeleteClick(agent);
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )
    }
  ];

  // Create agent button
  const createAgentButton = (
    <Button
      variant="contained"
      color="primary"
      startIcon={<AddIcon />}
      onClick={() => navigate('/agents/new')}
    >
      Create Agent
    </Button>
  );

  // If there are no agents and not loading, show empty state
  if (!isLoading && !error && agents.length === 0) {
    return (
      <EmptyState
        title="No Agents Found"
        message="Get started by creating your first agent."
        actionLabel="Create Agent"
        onAction={() => navigate('/agents/new')}
      />
    );
  }

  return (
    <>
      <Box sx={{ mb: 2 }}>
        <SearchBar
          placeholder="Search agents..."
          onSearch={handleSearch}
        />
      </Box>

      <DataTable
        columns={columns}
        data={filteredAgents}
        keyExtractor={(agent) => agent.id}
        isLoading={isLoading}
        error={error}
        onRowClick={handleViewAgent}
        refreshable
        onRefresh={onRefresh}
        actions={createAgentButton}
        emptyMessage="No agents found matching your search."
      />

      <ConfirmDialog
        open={deleteDialogOpen}
        title="Delete Agent"
        message={`Are you sure you want to delete the agent "${agentToDelete?.name}"? This action cannot be undone.`}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
        isLoading={isDeleting}
        confirmColor="error"
      />
    </>
  );
};

export default AgentList;
