import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone, FileRejection } from 'react-dropzone';
import { uploadVideo, processVideo } from '../services/api';
import { isValidVideoFile, isValidFileSize, formatFileSize } from '../utils/formatters';
import './VideoUploader.css';

const VideoUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setError(null);
    
    if (rejectedFiles.length > 0) {
      setError('Invalid file. Please upload a valid video file (MP4, MKV, MOV, AVI, WEBM).');
      return;
    }

    const selectedFile = acceptedFiles[0];
    
    // Validate file
    if (!isValidVideoFile(selectedFile)) {
      setError('Invalid file type. Please upload a video file.');
      return;
    }

    if (!isValidFileSize(selectedFile, 1024)) {
      setError('File size exceeds 1GB limit.');
      return;
    }

    setFile(selectedFile);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mkv', '.mov', '.avi', '.webm']
    },
    maxFiles: 1,
  });

  const handleUpload = async (): Promise<void> => {
    if (!file) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Upload file
      const uploadResponse = await uploadVideo(file, (progress: number) => {
        setUploadProgress(progress);
      });

      const jobId = uploadResponse.job_id || uploadResponse.id;
      if (!jobId) {
        throw new Error('No job ID received from server');
      }

      // Start processing
      await processVideo(jobId);

      // Navigate to processing status page
      navigate(`/process/${jobId}`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to upload video. Please try again.';
      setError(errorMessage);
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleRemove = (): void => {
    setFile(null);
    setError(null);
    setUploadProgress(0);
  };

  return (
    <div className="video-uploader">
      <h2>Upload Video</h2>
      <p className="uploader-description">
        Upload a video file to generate captions, transcripts, and summaries.
        Supported formats: MP4, MKV, MOV, AVI, WEBM (Max: 1GB)
      </p>

      {!file ? (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dropzone-content">
            <svg
              className="upload-icon"
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="dropzone-text">
              {isDragActive
                ? 'Drop the video file here'
                : 'Drag and drop a video file here, or click to select'}
            </p>
          </div>
        </div>
      ) : (
        <div className="file-info">
          <div className="file-details">
            <span className="file-name">{file.name}</span>
            <span className="file-size">{formatFileSize(file.size)}</span>
          </div>
          {isUploading && (
            <div className="upload-progress">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <span className="progress-text">{uploadProgress}%</span>
            </div>
          )}
          <div className="file-actions">
            {!isUploading && (
              <>
                <button onClick={handleRemove} className="btn btn-secondary">
                  Remove
                </button>
                <button onClick={handleUpload} className="btn btn-primary">
                  Upload & Process
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="error-message">
          <span>⚠️</span> {error}
        </div>
      )}
    </div>
  );
};

export default VideoUploader;

