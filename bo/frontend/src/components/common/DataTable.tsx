import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Paper,
  Typography,
  CircularProgress,
  Checkbox,
  IconButton,
  Tooltip,
  useTheme
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  FilterList as FilterListIcon
} from '@mui/icons-material';

// Define column type
export interface Column<T> {
  id: string;
  label: string;
  minWidth?: number;
  align?: 'left' | 'right' | 'center';
  format?: (value: any, row: T) => React.ReactNode;
  sortable?: boolean;
  hidden?: boolean;
}

// Define props
interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (item: T) => string | number;
  isLoading?: boolean;
  error?: string | null;
  selectable?: boolean;
  onSelectionChange?: (selectedItems: T[]) => void;
  onRowClick?: (item: T) => void;
  pagination?: boolean;
  defaultRowsPerPage?: number;
  rowsPerPageOptions?: number[];
  refreshable?: boolean;
  onRefresh?: () => void;
  emptyMessage?: string;
  actions?: React.ReactNode;
}

/**
 * DataTable component for displaying tabular data with sorting, pagination, and selection.
 */
function DataTable<T>({
  columns,
  data,
  keyExtractor,
  isLoading = false,
  error = null,
  selectable = false,
  onSelectionChange,
  onRowClick,
  pagination = true,
  defaultRowsPerPage = 10,
  rowsPerPageOptions = [5, 10, 25, 50],
  refreshable = false,
  onRefresh,
  emptyMessage = 'No data available',
  actions
}: DataTableProps<T>) {
  const theme = useTheme();
  
  // State for sorting
  const [orderBy, setOrderBy] = useState<string>('');
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');
  
  // State for pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(defaultRowsPerPage);
  
  // State for selection
  const [selected, setSelected] = useState<(string | number)[]>([]);

  // Handle sort request
  const handleRequestSort = (property: string) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  // Handle page change
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Handle select all click
  const handleSelectAllClick = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const newSelected = data.map(item => keyExtractor(item));
      setSelected(newSelected);
      onSelectionChange && onSelectionChange(data);
    } else {
      setSelected([]);
      onSelectionChange && onSelectionChange([]);
    }
  };

  // Handle row click
  const handleRowClick = (item: T) => {
    if (onRowClick) {
      onRowClick(item);
    }
  };

  // Handle checkbox click
  const handleCheckboxClick = (event: React.MouseEvent<HTMLButtonElement>, id: string | number) => {
    event.stopPropagation();
    const selectedIndex = selected.indexOf(id);
    let newSelected: (string | number)[] = [];

    if (selectedIndex === -1) {
      newSelected = [...selected, id];
    } else {
      newSelected = selected.filter(itemId => itemId !== id);
    }

    setSelected(newSelected);
    
    if (onSelectionChange) {
      const selectedItems = data.filter(item => newSelected.includes(keyExtractor(item)));
      onSelectionChange(selectedItems);
    }
  };

  // Check if an item is selected
  const isSelected = (id: string | number) => selected.indexOf(id) !== -1;

  // Sort function
  const sortData = (data: T[]) => {
    if (!orderBy) return data;
    
    return [...data].sort((a: any, b: any) => {
      const aValue = a[orderBy];
      const bValue = b[orderBy];
      
      if (aValue === bValue) return 0;
      
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return order === 'asc' 
          ? aValue.localeCompare(bValue) 
          : bValue.localeCompare(aValue);
      }
      
      return order === 'asc' 
        ? (aValue < bValue ? -1 : 1) 
        : (bValue < aValue ? -1 : 1);
    });
  };

  // Get visible data (sorted and paginated)
  const visibleData = pagination 
    ? sortData(data).slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
    : sortData(data);

  // Render loading state
  if (isLoading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  // Render empty state
  if (data.length === 0) {
    return (
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <Box sx={{ p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column' }}>
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            {emptyMessage}
          </Typography>
          {refreshable && onRefresh && (
            <IconButton onClick={onRefresh} color="primary">
              <RefreshIcon />
            </IconButton>
          )}
        </Box>
      </Paper>
    );
  }

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      {/* Table header with actions */}
      {(refreshable || actions) && (
        <Box sx={{ p: 1, display: 'flex', justifyContent: 'flex-end', borderBottom: `1px solid ${theme.palette.divider}` }}>
          {actions}
          {refreshable && onRefresh && (
            <Tooltip title="Refresh">
              <IconButton onClick={onRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      )}
      
      {/* Table */}
      <TableContainer>
        <Table stickyHeader aria-label="data table">
          <TableHead>
            <TableRow>
              {/* Selection column */}
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={selected.length > 0 && selected.length < data.length}
                    checked={data.length > 0 && selected.length === data.length}
                    onChange={handleSelectAllClick}
                    inputProps={{ 'aria-label': 'select all' }}
                  />
                </TableCell>
              )}
              
              {/* Data columns */}
              {columns.filter(column => !column.hidden).map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align || 'left'}
                  style={{ minWidth: column.minWidth }}
                  sortDirection={orderBy === column.id ? order : false}
                >
                  {column.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === column.id}
                      direction={orderBy === column.id ? order : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {visibleData.map((row) => {
              const id = keyExtractor(row);
              const isItemSelected = isSelected(id);
              
              return (
                <TableRow
                  hover
                  onClick={() => handleRowClick(row)}
                  role="checkbox"
                  aria-checked={isItemSelected}
                  tabIndex={-1}
                  key={id}
                  selected={isItemSelected}
                  sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
                >
                  {/* Selection checkbox */}
                  {selectable && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={isItemSelected}
                        onClick={(event) => handleCheckboxClick(event, id)}
                      />
                    </TableCell>
                  )}
                  
                  {/* Data cells */}
                  {columns.filter(column => !column.hidden).map((column) => {
                    const value = (row as any)[column.id];
                    return (
                      <TableCell key={column.id} align={column.align || 'left'}>
                        {column.format ? column.format(value, row) : value}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      
      {/* Pagination */}
      {pagination && (
        <TablePagination
          rowsPerPageOptions={rowsPerPageOptions}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      )}
    </Paper>
  );
}

export default DataTable;
