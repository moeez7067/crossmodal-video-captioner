import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getStatus, ProcessingStatus as ProcessingStatusType } from '../services/api';
import { formatDuration } from '../utils/formatters';
import './ProcessingStatus.css';

const ProcessingStatus: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [status, setStatus] = useState<ProcessingStatusType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!jobId) {
      setError('No job ID provided');
      setLoading(false);
      return;
    }

    const pollStatus = async (): Promise<void> => {
      try {
        const response = await getStatus(jobId);
        setStatus(response);
        setLoading(false);

        // If processing is complete, navigate to results
        if (response.status === 'completed') {
          setTimeout(() => {
            navigate(`/results/${jobId}`);
          }, 2000);
        } else if (response.status === 'failed') {
          setError(response.error || 'Processing failed');
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch status';
        setError(errorMessage);
        setLoading(false);
      }
    };

    // Poll immediately
    pollStatus();

    // Set up polling interval (every 2 seconds)
    const interval = setInterval(pollStatus, 2000);

    return () => clearInterval(interval);
  }, [jobId, navigate]);

  if (loading) {
    return (
      <div className="processing-status">
        <div className="loading-spinner">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="processing-status">
        <div className="error-message">
          <span>⚠️</span> {error}
        </div>
        <button onClick={() => navigate('/')} className="btn btn-primary">
          Go Back
        </button>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="processing-status">
        <div className="error-message">No status available</div>
      </div>
    );
  }

  const progress = status.progress || 0;
  const currentStatus = status.status || 'unknown';
  const stage = status.stage || 'Initializing';
  const estimatedTime = status.estimated_time_remaining;

  return (
    <div className="processing-status">
      <h2>Processing Video</h2>
      <p className="job-id">Job ID: {jobId}</p>

      <div className="status-card">
        <div className="status-header">
          <span className="status-label">Status:</span>
          <span className={`status-badge status-${currentStatus}`}>
            {currentStatus}
          </span>
        </div>

        <div className="progress-section">
          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="progress-info">
            <span className="progress-percentage">{Math.round(progress)}%</span>
            {estimatedTime && (
              <span className="estimated-time">
                Estimated time remaining: {formatDuration(estimatedTime)}
              </span>
            )}
          </div>
        </div>

        <div className="stage-info">
          <span className="stage-label">Current Stage:</span>
          <span className="stage-value">{stage}</span>
        </div>

        {currentStatus === 'completed' && (
          <div className="completion-message">
            ✅ Processing complete! Redirecting to results...
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingStatus;

