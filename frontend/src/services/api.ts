/**
 * API service for communicating with the backend.
 */

import axios, { AxiosInstance, AxiosProgressEvent } from 'axios';

// Base URL configuration - can be set via environment variable
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Types
export interface UploadResponse {
  job_id: string;
  id?: string;
  message?: string;
}

export interface ProcessingStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage?: string;
  estimated_time_remaining?: number;
  error?: string;
}

export interface Caption {
  text: string;
  start_time: number;
  end_time: number;
}

export interface Results {
  job_id: string;
  captions?: Caption[];
  captions_text?: string;
  transcript_text?: string;
  summary_text?: string;
  key_points?: string[];
  video_url?: string;
}

export interface ApiError {
  message: string;
  status: number;
  data?: unknown;
}

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);
      const apiError: ApiError = {
        message: error.response.data?.message || error.response.data?.detail || 'An error occurred',
        status: error.response.status,
        data: error.response.data,
      };
      return Promise.reject(apiError);
    } else if (error.request) {
      // Request made but no response received
      console.error('Network Error:', error.request);
      const apiError: ApiError = {
        message: 'Network error. Please check your connection.',
        status: 0,
      };
      return Promise.reject(apiError);
    } else {
      // Something else happened
      console.error('Error:', error.message);
      const apiError: ApiError = {
        message: error.message || 'An unexpected error occurred',
        status: 0,
      };
      return Promise.reject(apiError);
    }
  }
);

/**
 * Upload video file to the server.
 */
export const uploadVideo = async (
  file: File,
  onUploadProgress?: (progress: number) => void
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<UploadResponse>('/video/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent: AxiosProgressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onUploadProgress(percentCompleted);
      }
    },
  });

  return response.data;
};

/**
 * Start processing a video.
 */
export const processVideo = async (jobId: string): Promise<ProcessingStatus> => {
  const response = await apiClient.post<ProcessingStatus>(`/video/process`, { job_id: jobId });
  return response.data;
};

/**
 * Get processing status.
 */
export const getStatus = async (jobId: string): Promise<ProcessingStatus> => {
  const response = await apiClient.get<ProcessingStatus>(`/video/status/${jobId}`);
  return response.data;
};

/**
 * Get processing results.
 */
export const getResults = async (jobId: string): Promise<Results> => {
  const response = await apiClient.get<Results>(`/video/results/${jobId}`);
  return response.data;
};

/**
 * Download generated file.
 */
export const downloadFile = async (
  jobId: string,
  format: string,
  type: string = 'captions'
): Promise<Blob> => {
  const response = await apiClient.get(`/video/download/${jobId}/${format}`, {
    params: { type },
    responseType: 'blob',
  });
  return response.data;
};

/**
 * Health check endpoint.
 */
export const healthCheck = async (): Promise<{ status: string }> => {
  const response = await apiClient.get<{ status: string }>('/health');
  return response.data;
};

/**
 * Get list of processed videos (history).
 */
export interface ProcessedVideo {
  job_id: string;
  video_name: string;
  video_path: string;
  processed_date?: string;
  duration?: number;
  has_captions: boolean;
  has_transcript: boolean;
  has_summary: boolean;
  outputs_dir: string;
}

export interface HistoryResponse {
  videos: ProcessedVideo[];
}

export const getHistory = async (): Promise<HistoryResponse> => {
  const response = await apiClient.get<HistoryResponse>('/video/history');
  return response.data;
};

export default apiClient;

