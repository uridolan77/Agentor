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

// Define Tool type based on the backend schema
export interface Tool {
  id: number;
  name: string;
  description: string | null;
  tool_type: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  creator_id: number;
  team_id: number | null;
  configuration: Record<string, any> | string;
}

interface ToolListProps {
  tools: Tool[];
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
  onDelete: (id: number) => Promise<void>;
}

/**
 * ToolList component for displaying a list of tools.
 * Includes search, filtering, and actions (view, edit, delete).
 */
const ToolList: React.FC<ToolListProps> = ({
  tools,
  isLoading,
  error,
  onRefresh,
  onDelete
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTools, setFilteredTools] = useState<Tool[]>(tools);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [toolToDelete, setToolToDelete] = useState<Tool | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Filter tools based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredTools(tools);
      return;
    }

    const query = searchQuery.toLowerCase();
    const filtered = tools.filter(tool => 
      tool.name.toLowerCase().includes(query) || 
      (tool.description && tool.description.toLowerCase().includes(query)) ||
      tool.tool_type.toLowerCase().includes(query)
    );
    
    setFilteredTools(filtered);
  }, [tools, searchQuery]);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  // Handle view tool
  const handleViewTool = useCallback((tool: Tool) => {
    navigate(`/tools/${tool.id}`);
  }, [navigate]);

  // Handle edit tool
  const handleEditTool = useCallback((tool: Tool) => {
    navigate(`/tools/${tool.id}/edit`);
  }, [navigate]);

  // Handle delete tool
  const handleDeleteClick = useCallback((tool: Tool) => {
    setToolToDelete(tool);
    setDeleteDialogOpen(true);
  }, []);

  // Confirm delete tool
  const handleConfirmDelete = useCallback(async () => {
    if (!toolToDelete) return;
    
    setIsDeleting(true);
    try {
      await onDelete(toolToDelete.id);
      setDeleteDialogOpen(false);
      setToolToDelete(null);
    } catch (error) {
      console.error('Error deleting tool:', error);
    } finally {
      setIsDeleting(false);
    }
  }, [toolToDelete, onDelete]);

  // Cancel delete tool
  const handleCancelDelete = useCallback(() => {
    setDeleteDialogOpen(false);
    setToolToDelete(null);
  }, []);

  // Define table columns
  const columns: Column<Tool>[] = [
    {
      id: 'name',
      label: 'Name',
      minWidth: 150,
      sortable: true
    },
    {
      id: 'tool_type',
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
      format: (_, tool) => (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Tooltip title="View">
            <IconButton 
              size="small" 
              onClick={(e) => {
                e.stopPropagation();
                handleViewTool(tool);
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
                handleEditTool(tool);
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
                handleDeleteClick(tool);
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )
    }
  ];

  // Create tool button
  const createToolButton = (
    <Button
      variant="contained"
      color="primary"
      startIcon={<AddIcon />}
      onClick={() => navigate('/tools/new')}
    >
      Create Tool
    </Button>
  );

  // If there are no tools and not loading, show empty state
  if (!isLoading && !error && tools.length === 0) {
    return (
      <EmptyState
        title="No Tools Found"
        message="Get started by creating your first tool."
        actionLabel="Create Tool"
        onAction={() => navigate('/tools/new')}
      />
    );
  }

  return (
    <>
      <Box sx={{ mb: 2 }}>
        <SearchBar
          placeholder="Search tools..."
          onSearch={handleSearch}
        />
      </Box>

      <DataTable
        columns={columns}
        data={filteredTools}
        keyExtractor={(tool) => tool.id}
        isLoading={isLoading}
        error={error}
        onRowClick={handleViewTool}
        refreshable
        onRefresh={onRefresh}
        actions={createToolButton}
        emptyMessage="No tools found matching your search."
      />

      <ConfirmDialog
        open={deleteDialogOpen}
        title="Delete Tool"
        message={`Are you sure you want to delete the tool "${toolToDelete?.name}"? This action cannot be undone.`}
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

export default ToolList;
