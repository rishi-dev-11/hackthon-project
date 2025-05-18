import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if it exists
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auth API calls
export const loginUser = async (email, password) => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);
  
  const response = await axios.post(`${API_BASE_URL}/auth/login`, formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
  return response.data;
};

export const registerUser = async (userData) => {
  const response = await api.post('/auth/register', userData);
  return response.data;
};

export const getCurrentUser = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

export const logoutUser = () => {
  localStorage.removeItem('token');
};

// Document API calls
export const uploadDocument = async (formData) => {
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const formatDocument = async (documentId, templateId) => {
  const formData = new FormData();
  formData.append('document_id', documentId);
  formData.append('template_id', templateId);
  
  const response = await api.post('/format', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getPreview = async (documentId) => {
  const response = await api.get(`/preview/${documentId}`);
  return response.data;
};

export const downloadDocument = async (documentId) => {
  const response = await api.get(`/download/${documentId}`, {
    responseType: 'blob',
  });
  return response.data;
};

// User profile and settings
export const updateUserProfile = async (userData) => {
  const response = await api.post('/auth/update-profile', userData);
  return response.data;
};

export const updateUserSubscription = async (plan) => {
  const response = await api.post('/auth/upgrade', { plan });
  return response.data;
};

// Template API calls
export const getTemplates = async () => {
  const response = await api.get('/templates');
  return response.data;
};

export const getTemplate = async (templateId) => {
  const response = await api.get(`/templates/${templateId}`);
  return response.data;
};

// Add dev user to MongoDB (helper function for setup)
export const addDevUser = async () => {
  try {
    const userData = {
      email: 'mandarak123@gmail.com',
      password: 'Mak@1944',
      name: 'Developer',
      isDevMode: true
    };
    
    await registerUser(userData);
    return true;
  } catch (error) {
    console.error('Failed to add dev user:', error);
    return false;
  }
};

// Handle API errors globally
const handleError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    console.error('API Error:', error.response.data);
    throw new Error(error.response.data.detail || 'An error occurred');
  } else if (error.request) {
    // The request was made but no response was received
    console.error('Network Error:', error.request);
    throw new Error('Network error - please check your connection');
  } else {
    // Something happened in setting up the request that triggered an Error
    console.error('Error:', error.message);
    throw new Error('An error occurred');
  }
};

// Add error handling to all API calls
Object.keys(api).forEach(method => {
  const originalMethod = api[method];
  api[method] = async (...args) => {
    try {
      return await originalMethod(...args);
    } catch (error) {
      return handleError(error);
    }
  };
});