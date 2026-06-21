import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getResults, downloadFile, Results, Caption } from '../services/api';
import { copyToClipboard, downloadBlob } from '../utils/formatters';
import VideoPlayer from './VideoPlayer';
import HistorySidebar from './HistorySidebar';
import './ResultsViewer.css';

// Get API base URL for constructing full video URLs
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

type TabType = 'captions' | 'transcript' | 'summary';
type FormatType = 'srt' | 'vtt' | 'txt' | 'docx' | 'pdf';

interface FormatSelection {
  captions: 'srt' | 'vtt';
  transcript: 'txt' | 'docx';
  summary: 'txt' | 'pdf';
}

const ResultsViewer: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('captions');
  const [selectedFormat, setSelectedFormat] = useState<FormatSelection>({
    captions: 'srt',
    transcript: 'txt',
    summary: 'txt',
  });
  const [copySuccess, setCopySuccess] = useState<boolean>(false);

  useEffect(() => {
    if (!jobId) {
      setError('No job ID provided');
      setLoading(false);
      return;
    }

    const fetchResults = async (): Promise<void> => {
      try {
        const response = await getResults(jobId);
        setResults(response);
        setLoading(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch results';
        setError(errorMessage);
        setLoading(false);
      }
    };

    fetchResults();
  }, [jobId]);

  const handleDownload = async (type: string, format: FormatType): Promise<void> => {
    if (!jobId) return;

    try {
      const blob = await downloadFile(jobId, format, type);
      const filename = `${jobId}_${type}.${format}`;
      downloadBlob(blob, filename);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to download';
      alert(`Failed to download: ${errorMessage}`);
    }
  };

  const handleCopy = async (text: string | undefined): Promise<void> => {
    if (!text) {
      alert('No text to copy');
      return;
    }

    const success = await copyToClipboard(text);
    if (success) {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } else {
      alert('Failed to copy to clipboard');
    }
  };

  if (loading) {
    return (
      <div className="results-viewer">
        <div className="loading-spinner">Loading results...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-viewer">
        <div className="error-message">
          <span>⚠️</span> {error}
        </div>
        <button onClick={() => navigate('/')} className="btn btn-primary">
          Go Back
        </button>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-viewer">
        <div className="error-message">No results available</div>
      </div>
    );
  }

  return (
    <div className="results-viewer">
      <div className="results-header">
        <div className="results-header-content">
          <h2>Processing Results</h2>
          <p className="job-id">Job ID: {jobId}</p>
        </div>
        <div className="results-header-right">
          <button onClick={() => navigate('/')} className="btn btn-secondary">
            Process Another Video
          </button>
        </div>
      </div>

      {/* Video Player */}
      {results.video_url && (
        <div className="video-section">
          <VideoPlayer 
            videoUrl={
              results.video_url.startsWith('http') 
                ? results.video_url 
                : `${API_BASE_URL.replace('/api', '')}${results.video_url}`
            } 
            captions={results.captions || []} 
          />
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'captions' ? 'active' : ''}`}
          onClick={() => setActiveTab('captions')}
        >
          Captions
        </button>
        <button
          className={`tab ${activeTab === 'transcript' ? 'active' : ''}`}
          onClick={() => setActiveTab('transcript')}
        >
          Transcript
        </button>
        <button
          className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          Summary
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'captions' && (
          <div className="content-panel">
            <div className="panel-header">
              <h3>Captions</h3>
              <div className="format-selector">
                <label>Format:</label>
                <select
                  value={selectedFormat.captions}
                  onChange={(e) =>
                    setSelectedFormat({ ...selectedFormat, captions: e.target.value as 'srt' | 'vtt' })
                  }
                >
                  <option value="srt">SRT</option>
                  <option value="vtt">VTT</option>
                </select>
                <button
                  className="btn btn-primary"
                  onClick={() => handleDownload('captions', selectedFormat.captions)}
                >
                  Download
                </button>
              </div>
            </div>
            <div className="content-preview">
              <pre>{results.captions_text || 'No captions available'}</pre>
            </div>
          </div>
        )}

        {activeTab === 'transcript' && (
          <div className="content-panel">
            <div className="panel-header">
              <h3>Transcript</h3>
              <div className="format-selector">
                <label>Format:</label>
                <select
                  value={selectedFormat.transcript}
                  onChange={(e) =>
                    setSelectedFormat({ ...selectedFormat, transcript: e.target.value as 'txt' | 'docx' })
                  }
                >
                  <option value="txt">TXT</option>
                  <option value="docx">DOCX</option>
                </select>
                <button
                  className="btn btn-primary"
                  onClick={() => handleDownload('transcript', selectedFormat.transcript)}
                >
                  Download
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => handleCopy(results.transcript_text)}
                >
                  {copySuccess ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
            <div className="content-preview">
              <pre>{results.transcript_text || 'No transcript available'}</pre>
            </div>
          </div>
        )}

        {activeTab === 'summary' && (
          <div className="content-panel">
            <div className="panel-header">
              <h3>Summary</h3>
              <div className="format-selector">
                <label>Format:</label>
                <select
                  value={selectedFormat.summary}
                  onChange={(e) =>
                    setSelectedFormat({ ...selectedFormat, summary: e.target.value as 'txt' | 'pdf' })
                  }
                >
                  <option value="txt">TXT</option>
                  <option value="pdf">PDF</option>
                </select>
                <button
                  className="btn btn-primary"
                  onClick={() => handleDownload('summary', selectedFormat.summary)}
                >
                  Download
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => handleCopy(results.summary_text)}
                >
                  {copySuccess ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
            <div className="content-preview summary-preview">
              {results.key_points && results.key_points.length > 0 && (
                <div className="key-points-section">
                  <h4 className="section-title">Key Points</h4>
                  <ul className="key-points-list">
                    {(() => {
                      // Handle key points - split if they come as a single string
                      let points = results.key_points || [];
                      if (points.length === 1 && points[0].includes('. ')) {
                        // If single point contains multiple sentences, try to split intelligently
                        const singlePoint = points[0];
                        // Split by sentence endings, but also look for patterns like "speaker: " or numbered patterns
                        const splitPoints = singlePoint
                          .split(/(?<=[.!?])\s+(?=[A-Z]|roland|martin)/i)
                          .map(p => p.trim())
                          .filter(p => p.length > 0 && p.length > 20); // Filter out very short fragments
                        if (splitPoints.length > 1) {
                          points = splitPoints;
                        }
                      }
                      return points.map((point, index) => (
                        <li key={index} className="key-point-item">
                          <span className="key-point-bullet">•</span>
                          <span className="key-point-text">{point.trim()}</span>
                        </li>
                      ));
                    })()}
                  </ul>
                </div>
              )}
              <div className="summary-section">
                <h4 className="section-title">Full Summary</h4>
                <div className="summary-text">
                  {results.summary_text ? (
                    <p className="summary-paragraph-full">
                      {results.summary_text}
                    </p>
                  ) : (
                    <p className="no-content">No summary available</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsViewer;

