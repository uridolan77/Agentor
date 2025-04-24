import axios from 'axios';

// Configure Axios to use the backend server URL
axios.defaults.baseURL = 'http://localhost:9000';

// Add a request interceptor for handling errors and authentication
axios.interceptors.request.use(
  (config) => {
    // Add authentication token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor for handling errors
axios.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle 401 Unauthorized errors
    if (error.response && error.response.status === 401) {
      console.log('Authentication token expired or invalid. Logging out...');

      // Clear auth data from localStorage
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user');

      // Remove Authorization header
      delete axios.defaults.headers.common['Authorization'];

      // Redirect to login page
      window.location.href = '/login';
    }

    // Handle other errors
    return Promise.reject(error);
  }
);

export default axios;
