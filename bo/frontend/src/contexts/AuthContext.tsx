import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from '../utils/axios'; // Using our configured axios instance

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
  token: string | null;
}

interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  full_name?: string;
}

interface AuthContextType extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
  hasPermission: (permission: string) => boolean;
}

const initialAuthState: AuthState = {
  isAuthenticated: false,
  isLoading: true,
  user: null,
  token: null,
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [auth, setAuth] = useState<AuthState>(initialAuthState);

  // Initialize auth state from localStorage
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('user');

    if (storedToken && storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setAuth({
          isAuthenticated: true,
          isLoading: false,
          user,
          token: storedToken,
        });

        // Set default Authorization header
        axios.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`;
      } catch (error) {
        console.error('Error parsing stored user:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
        setAuth({ ...initialAuthState, isLoading: false });
      }
    } else {
      setAuth({ ...initialAuthState, isLoading: false });
    }
  }, []);

  const login = async (username: string, password: string): Promise<void> => {
    try {
      setAuth(prev => ({ ...prev, isLoading: true }));

      // Use the correct endpoint and format (form data instead of JSON)
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await axios.post('/auth/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      const { access_token, user_id, username: userName, role } = response.data;

      // Create user object from response data
      const user = {
        id: user_id,
        username: userName,
        email: '', // Add this if available in the response
        role: role,
      };

      // Store in localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user', JSON.stringify(user));

      // Set default Authorization header
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

      setAuth({
        isAuthenticated: true,
        isLoading: false,
        user,
        token: access_token,
      });
    } catch (error) {
      console.error('Login error:', error);
      setAuth(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  };

  const logout = (): void => {
    // Remove from localStorage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');

    // Remove Authorization header
    delete axios.defaults.headers.common['Authorization'];

    setAuth({
      isAuthenticated: false,
      isLoading: false,
      user: null,
      token: null,
    });
  };

  const refreshToken = async (): Promise<boolean> => {
    try {
      setAuth(prev => ({ ...prev, isLoading: true }));

      const response = await axios.post('/api/auth/refresh');
      const { access_token, user } = response.data;

      // Store in localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('user', JSON.stringify(user));

      // Set default Authorization header
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

      setAuth({
        isAuthenticated: true,
        isLoading: false,
        user,
        token: access_token,
      });

      return true;
    } catch (error) {
      console.error('Token refresh error:', error);
      logout();
      return false;
    }
  };

  const hasPermission = (permission: string): boolean => {
    // If user is admin, they have all permissions
    if (auth.user?.role === 'admin') {
      return true;
    }

    // Add more specific permission logic here if needed
    // For now, we'll just check if the user is authenticated
    return auth.isAuthenticated;
  };

  const value: AuthContextType = {
    ...auth,
    login,
    logout,
    refreshToken,
    hasPermission,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;
