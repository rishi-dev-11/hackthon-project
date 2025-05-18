import axios from 'axios';

// Configure Axios base URL based on environment
const isProduction = import.meta.env.PROD;
const API_URL = isProduction ? '' : 'http://localhost:8000';

axios.defaults.baseURL = API_URL;
axios.defaults.withCredentials = true; // Enable cookies for CORS

// Request interceptor for adding auth token
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor to handle errors globally
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Handle authentication errors
      if (error.response.status === 401) {
        localStorage.removeItem('token');
        // Optionally redirect to login
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
      }
      
      // Format error message from server
      const errorMessage = error.response.data?.detail || 'An unexpected error occurred';
      return Promise.reject(new Error(errorMessage));
    }
    return Promise.reject(error);
  }
);

// Document Upload
export const uploadDocument = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

// Get Templates
export const getTemplates = async () => {
  try {
    const response = await axios.get('/api/templates');
    return response.data;
  } catch (error) {
    console.error('Error fetching templates:', error);
    throw error;
  }
};

// Format Document
export const formatDocument = async (documentId, templateId) => {
  try {
    const formData = new FormData();
    formData.append('document_id', documentId);
    formData.append('template_id', templateId);
    
    const response = await axios.post('/api/format', formData);
    return response.data;
  } catch (error) {
    console.error('Error formatting document:', error);
    throw error;
  }
};

// Get Document Preview
export const getPreview = async (documentId) => {
  try {
    const response = await axios.get(`/api/preview/${documentId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching preview:', error);
    throw error;
  }
};

// Download Document
export const downloadDocument = async (documentId) => {
  try {
    const response = await axios.get(`/api/download/${documentId}`, {
      responseType: 'blob',
    });
    
    // Create a URL for the blob and trigger download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    
    // Get filename from Content-Disposition header if available
    const contentDisposition = response.headers['content-disposition'];
    let filename = 'document.docx';
    
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="(.+)"/);
      if (filenameMatch && filenameMatch.length === 2) {
        filename = filenameMatch[1];
      }
    }
    
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    
    return { success: true, filename };
  } catch (error) {
    console.error('Error downloading document:', error);
    throw error;
  }
};

// Delete Document
export const deleteDocument = async (documentId) => {
  try {
    const response = await axios.delete(`/api/document/${documentId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting document:', error);
    throw error;
  }
};

// Authentication APIs
export const loginUser = async (email, password) => {
  try {
    // Using URLSearchParams as FastAPI expects form-urlencoded for OAuth
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await axios.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    
    if (response.data && response.data.access_token) {
      localStorage.setItem('token', response.data.access_token);
      
      // Also store user data for quick access
      if (response.data.user) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
      }
    }
    
    return response.data;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

export const registerUser = async (userData) => {
  try {
    // Convert to FormData if it's not already
    let data = userData;
    if (!(userData instanceof FormData)) {
      data = new FormData();
      Object.keys(userData).forEach(key => {
        data.append(key, userData[key]);
      });
    }
    
    const response = await axios.post('/api/auth/register', data);
    return response.data;
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

export const logoutUser = () => {
  localStorage.removeItem('token');
};

export const getCurrentUser = async () => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      // No token, so no user is authenticated
      return null; 
    }
    const response = await axios.get('/api/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    });
    
    // Ensure that the response includes subscription and isDevMode
    return {
      ...response.data,
      subscription: response.data.subscription || 'free', // Default to free if not present
      isDevMode: response.data.isDevMode || false // Default to false if not present
    };
  } catch (error) {
    console.error('Error fetching current user:', error.response ? error.response.data : error.message);
    // If error (e.g., token expired), treat as logged out
    localStorage.removeItem('token'); 
    return null;
  }
};

// Update User Subscription
export const updateUserSubscription = async (newTier) => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await axios.put('/api/user/subscription', 
      { newTier }, 
      { headers: { Authorization: `Bearer ${token}` } }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error updating subscription:', error);
    throw error;
  }
}; 