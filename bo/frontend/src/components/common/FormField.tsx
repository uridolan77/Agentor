import React from 'react';
import {
  TextField,
  FormControl,
  FormHelperText,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Switch,
  FormGroup,
  Radio,
  RadioGroup,
  Autocomplete,
  Chip,
  Box,
  Typography,
  SxProps,
  Theme,
  SelectChangeEvent
} from '@mui/material';

// Define field types
export type FieldType = 
  | 'text' 
  | 'number' 
  | 'email' 
  | 'password' 
  | 'select' 
  | 'multiselect' 
  | 'checkbox' 
  | 'switch' 
  | 'radio' 
  | 'textarea' 
  | 'date' 
  | 'time' 
  | 'datetime';

// Define option type for select, multiselect, and radio fields
export interface FieldOption {
  value: string | number | boolean;
  label: string;
}

// Define props
interface FormFieldProps {
  type: FieldType;
  name: string;
  label: string;
  value: any;
  onChange: (name: string, value: any) => void;
  options?: FieldOption[];
  error?: string;
  required?: boolean;
  disabled?: boolean;
  placeholder?: string;
  helperText?: string;
  fullWidth?: boolean;
  multiline?: boolean;
  rows?: number;
  min?: number;
  max?: number;
  step?: number;
  sx?: SxProps<Theme>;
}

/**
 * FormField component for standardized form fields.
 * Supports various field types and provides consistent styling and behavior.
 */
const FormField: React.FC<FormFieldProps> = ({
  type,
  name,
  label,
  value,
  onChange,
  options = [],
  error,
  required = false,
  disabled = false,
  placeholder,
  helperText,
  fullWidth = true,
  multiline = false,
  rows = 4,
  min,
  max,
  step,
  sx
}) => {
  // Handle change for different field types
  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | SelectChangeEvent<any>
  ) => {
    let newValue;
    
    if ('checked' in event.target) {
      // For checkbox and switch
      newValue = event.target.checked;
    } else if (type === 'number') {
      // For number inputs
      newValue = event.target.value === '' ? '' : Number(event.target.value);
    } else {
      // For all other inputs
      newValue = event.target.value;
    }
    
    onChange(name, newValue);
  };

  // Handle change for multiselect
  const handleMultiselectChange = (event: React.SyntheticEvent, newValue: any) => {
    onChange(name, newValue);
  };

  // Render field based on type
  const renderField = () => {
    switch (type) {
      case 'select':
        return (
          <FormControl 
            fullWidth={fullWidth} 
            error={!!error} 
            required={required}
            disabled={disabled}
            sx={sx}
          >
            <InputLabel id={`${name}-label`}>{label}</InputLabel>
            <Select
              labelId={`${name}-label`}
              id={name}
              name={name}
              value={value || ''}
              label={label}
              onChange={handleChange}
            >
              {options.map((option) => (
                <MenuItem 
                  key={option.value.toString()} 
                  value={typeof option.value === 'boolean' ? option.value.toString() : option.value}
                >
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            {(error || helperText) && (
              <FormHelperText>{error || helperText}</FormHelperText>
            )}
          </FormControl>
        );
        
      case 'multiselect':
        return (
          <FormControl 
            fullWidth={fullWidth} 
            error={!!error} 
            required={required}
            disabled={disabled}
            sx={sx}
          >
            <Autocomplete
              multiple
              id={name}
              options={options.map(option => option.value)}
              getOptionLabel={(option) => {
                const found = options.find(o => o.value === option);
                return found ? found.label : option.toString();
              }}
              value={value || []}
              onChange={handleMultiselectChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label={label}
                  placeholder={placeholder}
                  error={!!error}
                  helperText={error || helperText}
                />
              )}
              renderTags={(tagValue, getTagProps) =>
                tagValue.map((option, index) => {
                  const found = options.find(o => o.value === option);
                  return (
                    <Chip
                      label={found ? found.label : option.toString()}
                      {...getTagProps({ index })}
                    />
                  );
                })
              }
              disabled={disabled}
            />
          </FormControl>
        );
        
      case 'checkbox':
        return (
          <FormControl 
            fullWidth={fullWidth} 
            error={!!error} 
            required={required}
            disabled={disabled}
            sx={sx}
          >
            <FormControlLabel
              control={
                <Checkbox
                  id={name}
                  name={name}
                  checked={!!value}
                  onChange={handleChange}
                />
              }
              label={label}
            />
            {(error || helperText) && (
              <FormHelperText>{error || helperText}</FormHelperText>
            )}
          </FormControl>
        );
        
      case 'switch':
        return (
          <FormControl 
            fullWidth={fullWidth} 
            error={!!error} 
            required={required}
            disabled={disabled}
            sx={sx}
          >
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch
                    id={name}
                    name={name}
                    checked={!!value}
                    onChange={handleChange}
                  />
                }
                label={label}
              />
            </FormGroup>
            {(error || helperText) && (
              <FormHelperText>{error || helperText}</FormHelperText>
            )}
          </FormControl>
        );
        
      case 'radio':
        return (
          <FormControl 
            fullWidth={fullWidth} 
            error={!!error} 
            required={required}
            disabled={disabled}
            sx={sx}
          >
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {label}
            </Typography>
            <RadioGroup
              id={name}
              name={name}
              value={value || ''}
              onChange={handleChange}
            >
              {options.map((option) => (
                <FormControlLabel
                  key={option.value.toString()}
                  value={option.value}
                  control={<Radio />}
                  label={option.label}
                />
              ))}
            </RadioGroup>
            {(error || helperText) && (
              <FormHelperText>{error || helperText}</FormHelperText>
            )}
          </FormControl>
        );
        
      case 'textarea':
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            value={value || ''}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            placeholder={placeholder}
            fullWidth={fullWidth}
            multiline
            rows={rows}
            sx={sx}
          />
        );
        
      case 'number':
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            type="number"
            value={value}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            placeholder={placeholder}
            fullWidth={fullWidth}
            inputProps={{ min, max, step }}
            sx={sx}
          />
        );
        
      case 'date':
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            type="date"
            value={value || ''}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            fullWidth={fullWidth}
            InputLabelProps={{ shrink: true }}
            sx={sx}
          />
        );
        
      case 'time':
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            type="time"
            value={value || ''}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            fullWidth={fullWidth}
            InputLabelProps={{ shrink: true }}
            sx={sx}
          />
        );
        
      case 'datetime':
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            type="datetime-local"
            value={value || ''}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            fullWidth={fullWidth}
            InputLabelProps={{ shrink: true }}
            sx={sx}
          />
        );
        
      // Default to text field
      default:
        return (
          <TextField
            id={name}
            name={name}
            label={label}
            type={type}
            value={value || ''}
            onChange={handleChange}
            error={!!error}
            helperText={error || helperText}
            required={required}
            disabled={disabled}
            placeholder={placeholder}
            fullWidth={fullWidth}
            multiline={multiline}
            rows={multiline ? rows : undefined}
            sx={sx}
          />
        );
    }
  };

  return renderField();
};

export default FormField;
