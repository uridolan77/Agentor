import React, { useState, useMemo } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Paper,
  TablePagination,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  CloudDownload as DownloadIcon,
  ArrowDropDown as ArrowDropDownIcon
} from '@mui/icons-material';
import { visuallyHidden } from '@mui/utils';
import { ChartData, VisualizationConfig } from '../../types/reporting/reportTypes';

interface Column {
  name: string;
  type: string;
  label: string;
}

interface TableChartProps {
  data: ChartData[];
  config: VisualizationConfig;
}

type Order = 'asc' | 'desc';

interface EnhancedTableProps {
  columns: Column[];
  onRequestSort: (event: React.MouseEvent<unknown>, property: string) => void;
  order: Order;
  orderBy: string;
  formatHeader?: (name: string) => string;
}

function EnhancedTableHead(props: EnhancedTableProps) {
  const { columns, order, orderBy, onRequestSort, formatHeader } = props;

  const createSortHandler =
    (property: string) => (event: React.MouseEvent<unknown>) => {
      onRequestSort(event, property);
    };

  return (
    <TableHead>
      <TableRow>
        {columns.map((column) => (
          <TableCell
            key={column.name}
            sortDirection={orderBy === column.name ? order : false}
          >
            <TableSortLabel
              active={orderBy === column.name}
              direction={orderBy === column.name ? order : 'asc'}
              onClick={createSortHandler(column.name)}
            >
              {formatHeader ? formatHeader(column.label) : column.label}
              {orderBy === column.name ? (
                <Box component="span" sx={visuallyHidden}>
                  {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                </Box>
              ) : null}
            </TableSortLabel>
          </TableCell>
        ))}
      </TableRow>
    </TableHead>
  );
}

const TableChart: React.FC<TableChartProps> = ({ data, config }) => {
  // Extract settings from config
  const settings = config || {};
  const dimensions = config.dimensions || [];
  const metrics = config.metrics || [];

  // Convert data to the expected format
  const tableData = {
    columns: Object.keys(data[0] || {}).map(key => ({
      name: key,
      type: typeof data[0][key] === 'number' ? 'number' : 'string',
      label: key
    })),
    rows: data
  };
  // State for sorting, pagination, and filtering
  const [order, setOrder] = useState<Order>('asc');
  const [orderBy, setOrderBy] = useState<string>('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(settings.pageSize || 10);
  const [searchText, setSearchText] = useState('');
  const [columnFilter, setColumnFilter] = useState<string>('');

  // Get columns to display
  const displayColumns = useMemo(() => {
    if (settings.columnOrder) {
      return tableData.columns.filter(col => settings.columnOrder!.includes(col.name))
        .sort((a, b) =>
          settings.columnOrder!.indexOf(a.name) - settings.columnOrder!.indexOf(b.name)
        );
    }

    // If no specific order is provided, display dimensions first, then metrics
    const allColumns = [...dimensions, ...metrics];
    return tableData.columns.filter(col => allColumns.includes(col.name));
  }, [tableData.columns, dimensions, metrics, settings.columnOrder]);

  // Format the cell value based on column type
  const formatCellValue = (value: any, columnType: string, columnName: string) => {
    if (value === null || value === undefined) {
      return '-';
    }

    // Check if there's a specific formatter for this column
    const columnFormatter = settings.formatting && settings.formatting[columnName];

    if (columnFormatter) {
      return columnFormatter(value);
    }

    // Default formatting based on column type
    switch (columnType) {
      case 'number':
        return typeof value === 'number' ? value.toLocaleString() : value;

      case 'currency':
        return typeof value === 'number'
          ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
          : value;

      case 'percentage':
        return typeof value === 'number'
          ? `${(value * 100).toFixed(2)}%`
          : value;

      case 'date':
        return value instanceof Date || typeof value === 'string'
          ? new Date(value).toLocaleDateString()
          : value;

      case 'datetime':
        return value instanceof Date || typeof value === 'string'
          ? new Date(value).toLocaleString()
          : value;

      default:
        return value;
    }
  };

  // Format header text - convert camelCase or snake_case to Title Case
  const formatHeader = (text: string): string => {
    return text
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, (str) => str.toUpperCase());
  };

  // Filter and sort data
  const filteredSortedRows = useMemo(() => {
    // Filter the data
    let filteredData = tableData.rows;

    if (searchText) {
      const searchLower = searchText.toLowerCase();
      filteredData = filteredData.filter(row => {
        // If a specific column is selected, only search in that column
        if (columnFilter) {
          const cellValue = row[columnFilter];
          return cellValue !== null &&
            cellValue !== undefined &&
            String(cellValue).toLowerCase().includes(searchLower);
        }

        // Otherwise search all displayed columns
        return displayColumns.some(col => {
          const cellValue = row[col.name];
          return cellValue !== null &&
            cellValue !== undefined &&
            String(cellValue).toLowerCase().includes(searchLower);
        });
      });
    }

    // Sort the data
    if (orderBy) {
      filteredData = [...filteredData].sort((a, b) => {
        const aValue = a[orderBy];
        const bValue = b[orderBy];

        // Handle null/undefined values
        if (aValue === null || aValue === undefined) return order === 'asc' ? -1 : 1;
        if (bValue === null || bValue === undefined) return order === 'asc' ? 1 : -1;

        // Determine the column type
        const column = tableData.columns.find(col => col.name === orderBy);

        // Sort based on column type
        if (column && column.type === 'number') {
          return order === 'asc' ? Number(aValue) - Number(bValue) : Number(bValue) - Number(aValue);
        } else if (column && (column.type === 'date' || column.type === 'datetime')) {
          const dateA = new Date(aValue);
          const dateB = new Date(bValue);
          return order === 'asc' ? dateA.getTime() - dateB.getTime() : dateB.getTime() - dateA.getTime();
        } else {
          // String comparison for everything else
          const strA = String(aValue).toLowerCase();
          const strB = String(bValue).toLowerCase();
          return order === 'asc' ? strA.localeCompare(strB) : strB.localeCompare(strA);
        }
      });
    }

    return filteredData;
  }, [tableData.rows, tableData.columns, displayColumns, searchText, columnFilter, orderBy, order]);

  // Handle sort request
  const handleRequestSort = (
    event: React.MouseEvent<unknown>,
    property: string,
  ) => {
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

  // Handle search input change
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchText(event.target.value);
    setPage(0);  // Reset to first page when searching
  };

  // Handle column filter change
  const handleColumnFilterChange = (event: SelectChangeEvent) => {
    setColumnFilter(event.target.value);
    setPage(0);  // Reset to first page when changing filter
  };

  // Handle CSV download
  const handleDownloadCSV = () => {
    const headers = displayColumns.map(col => col.label).join(',');
    const rows = filteredSortedRows.map(row =>
      displayColumns.map(col => {
        // Format the cell value
        const value = formatCellValue(row[col.name], col.type, col.name);
        // Escape commas by wrapping in quotes
        return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
      }).join(',')
    ).join('\n');

    const csvContent = `${headers}\n${rows}`;
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${settings.title || 'data'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Calculate if we should show empty rows to fill the page
  const emptyRows = page > 0
    ? Math.max(0, (1 + page) * rowsPerPage - filteredSortedRows.length)
    : 0;

  // Get current page of data
  const paginatedRows = settings.showPagination !== false
    ? filteredSortedRows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
    : filteredSortedRows;

  return (
    <Box sx={{ width: '100%' }}>
      {/* Search and Filter Controls */}
      {settings.showSearch !== false && (
        <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
          <TextField
            size="small"
            label="Search"
            variant="outlined"
            value={searchText}
            onChange={handleSearchChange}
            sx={{ flexGrow: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
          />

          <FormControl sx={{ minWidth: 150 }} size="small">
            <InputLabel>Search in Column</InputLabel>
            <Select
              value={columnFilter}
              label="Search in Column"
              onChange={handleColumnFilterChange}
              endAdornment={
                <InputAdornment position="end">
                  <ArrowDropDownIcon />
                </InputAdornment>
              }
            >
              <MenuItem value="">
                <em>All Columns</em>
              </MenuItem>
              {displayColumns.map((column) => (
                <MenuItem key={column.name} value={column.name}>
                  {formatHeader(column.label)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {settings.showDownload !== false && (
            <Tooltip title="Download CSV">
              <IconButton onClick={handleDownloadCSV}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      )}

      <TableContainer component={Paper}>
        <Table sx={{ width: '100%' }} size="medium" aria-label="data table">
          <EnhancedTableHead
            columns={displayColumns}
            order={order}
            orderBy={orderBy}
            onRequestSort={handleRequestSort}
            formatHeader={formatHeader}
          />
          <TableBody>
            {paginatedRows.length > 0 ? (
              paginatedRows.map((row, index) => (
                <TableRow
                  hover
                  key={index}
                  sx={{ '&:nth-of-type(even)': { backgroundColor: 'rgba(0, 0, 0, 0.02)' } }}
                >
                  {displayColumns.map((column) => (
                    <TableCell
                      key={column.name}
                      align={column.type === 'number' || column.type === 'currency' || column.type === 'percentage' ? 'right' : 'left'}
                      sx={{
                        width: settings.columnWidths?.[column.name] ? `${settings.columnWidths[column.name]}px` : 'auto',
                      }}
                    >
                      {formatCellValue(row[column.name], column.type, column.name)}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={displayColumns.length} align="center">
                  <Typography variant="body1" sx={{ py: 5 }}>
                    No data to display
                  </Typography>
                </TableCell>
              </TableRow>
            )}

            {emptyRows > 0 && (
              <TableRow style={{ height: 53 * emptyRows }}>
                <TableCell colSpan={displayColumns.length} />
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination Controls */}
      {settings.showPagination !== false && (
        <TablePagination
          rowsPerPageOptions={[5, 10, 25, 50, 100]}
          component="div"
          count={filteredSortedRows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          labelRowsPerPage="Rows per page:"
          labelDisplayedRows={({ from, to, count }) => `${from}–${to} of ${count}`}
        />
      )}
    </Box>
  );
};

export default TableChart;