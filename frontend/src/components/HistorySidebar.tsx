import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory, ProcessedVideo } from '../services/api';
import './HistorySidebar.css';

interface HistorySidebarProps {
  isOpen: boolean;
  onClose: () => void;
  currentJobId?: string;
}

const HistorySidebar: React.FC<HistorySidebarProps> = ({ isOpen, onClose, currentJobId }) => {
  const [videos, setVideos] = useState<ProcessedVideo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (isOpen) {
      loadHistory();
    }
  }, [isOpen]);

  const loadHistory = async (): Promise<void> => {
    try {
      setLoading(true);
      const response = await getHistory();
      setVideos(response.videos);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load history';
      setError(errorMessage);
      console.error('Error loading history:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleVideoClick = (jobId: string): void => {
    navigate(`/results/${jobId}`);
    onClose();
  };

  const formatDate = (dateString?: string): string => {
    if (!dateString) return 'Unknown date';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return dateString;
    }
  };

  const formatDuration = (seconds?: number): string => {
    if (!seconds) return 'Unknown';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && <div className="history-overlay" onClick={onClose} />}
      
      {/* Sidebar */}
      <div className={`history-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="history-header">
          <h3>Processing History</h3>
          <button className="close-btn" onClick={onClose} aria-label="Close history">
            ×
          </button>
        </div>

        <div className="history-content">
          {loading && (
            <div className="history-loading">Loading history...</div>
          )}

          {error && (
            <div className="history-error">
              <span>⚠️</span> {error}
            </div>
          )}

          {!loading && !error && videos.length === 0 && (
            <div className="history-empty">
              No processed videos found.
            </div>
          )}

          {!loading && !error && videos.length > 0 && (
            <div className="history-list">
              {videos.map((video) => (
                <div
                  key={video.job_id}
                  className={`history-item ${video.job_id === currentJobId ? 'active' : ''}`}
                  onClick={() => handleVideoClick(video.job_id)}
                >
                  <div className="history-item-header">
                    <h4 className="history-item-title">{video.video_name}</h4>
                    <div className="history-item-badges">
                      {video.has_captions && <span className="badge">Captions</span>}
                      {video.has_transcript && <span className="badge">Transcript</span>}
                      {video.has_summary && <span className="badge">Summary</span>}
                    </div>
                  </div>
                  <div className="history-item-meta">
                    <span className="history-item-date">{formatDate(video.processed_date)}</span>
                    <span className="history-item-duration">{formatDuration(video.duration)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default HistorySidebar;

