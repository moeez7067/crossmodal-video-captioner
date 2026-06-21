import React, { useRef, useState, useEffect } from 'react';
import { Caption } from '../services/api';
import './VideoPlayer.css';

interface VideoPlayerProps {
  videoUrl?: string;
  captions?: Caption[];
  onSeek?: (time: number) => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl, captions = [], onSeek }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [currentCaption, setCurrentCaption] = useState<string>('');
  const [showCaptions, setShowCaptions] = useState<boolean>(true);
  const [showCaptionList, setShowCaptionList] = useState<boolean>(true); // New state for collapsible list
  const [volume, setVolume] = useState<number>(1.0);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = (): void => {
      setCurrentTime(video.currentTime);
      updateCurrentCaption(video.currentTime);
    };

    const updateDuration = (): void => {
      setDuration(video.duration);
    };

    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('loadedmetadata', updateDuration);

    return () => {
      video.removeEventListener('timeupdate', updateTime);
      video.removeEventListener('loadedmetadata', updateDuration);
    };
  }, []);

  useEffect(() => {
    updateCurrentCaption(currentTime);
  }, [currentTime, captions]);

  const updateCurrentCaption = (time: number): void => {
    if (!captions || captions.length === 0) {
      setCurrentCaption('');
      return;
    }

    const caption = captions.find(
      (cap) => time >= cap.start_time && time <= cap.end_time
    );

    setCurrentCaption(caption ? caption.text : '');
  };

  const handlePlayPause = (): void => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>): void => {
    const video = videoRef.current;
    if (!video || !video.duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const newTime = pos * video.duration;
    video.currentTime = newTime;
    setCurrentTime(newTime);
    if (onSeek) {
      onSeek(newTime);
    }
  };

  const handleCaptionClick = (caption: Caption): void => {
    const video = videoRef.current;
    if (!video) return;
    
    // Seek to the start of the clicked caption
    video.currentTime = caption.start_time;
    setCurrentTime(caption.start_time);
    if (onSeek) {
      onSeek(caption.start_time);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const video = videoRef.current;
    if (!video) return;
    const newVolume = parseFloat(e.target.value);
    video.volume = newVolume;
    setVolume(newVolume);
  };

  const handlePlaybackRateChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    const video = videoRef.current;
    if (!video) return;
    const newRate = parseFloat(e.target.value);
    video.playbackRate = newRate;
    setPlaybackRate(newRate);
  };

  const handleFullscreen = (): void => {
    const video = videoRef.current;
    if (!video) return;
    
    if (video.requestFullscreen) {
      video.requestFullscreen();
    } else if ((video as any).webkitRequestFullscreen) {
      (video as any).webkitRequestFullscreen();
    } else if ((video as any).mozRequestFullScreen) {
      (video as any).mozRequestFullScreen();
    }
  };

  const formatTime = (seconds: number): string => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  if (!videoUrl) {
    return <div className="video-player-placeholder">No video available</div>;
  }

  // Validate video URL
  const isValidUrl = videoUrl && (videoUrl.startsWith('http') || videoUrl.startsWith('/'));

  return (
    <div className="video-player-container">
      <div className="video-wrapper">
        <video
          ref={videoRef}
          src={isValidUrl ? videoUrl : undefined}
          className="video-element"
          onClick={handlePlayPause}
          onError={(e) => {
            console.error('Video load error:', e);
            const video = e.currentTarget;
            console.error('Video error details:', {
              error: video.error,
              networkState: video.networkState,
              readyState: video.readyState,
              src: video.src
            });
          }}
        />
        {showCaptions && currentCaption && (
          <div className="caption-overlay">{currentCaption}</div>
        )}
      </div>

      <div className="video-controls">
        <button
          className="control-btn"
          onClick={handlePlayPause}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        <div className="progress-container" onClick={handleSeek}>
          <div
            className="progress-bar"
            style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>

        <div className="time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        <div className="volume-control">
          <span className="volume-icon">🔊</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={handleVolumeChange}
            className="volume-slider"
            aria-label="Volume"
          />
        </div>

        <select
          className="playback-rate-select"
          value={playbackRate}
          onChange={handlePlaybackRateChange}
          aria-label="Playback rate"
        >
          <option value="0.5">0.5x</option>
          <option value="0.75">0.75x</option>
          <option value="1">1x</option>
          <option value="1.25">1.25x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2x</option>
        </select>

        <button
          className="control-btn"
          onClick={() => setShowCaptions(!showCaptions)}
          aria-label={showCaptions ? 'Hide captions' : 'Show captions'}
          title={showCaptions ? 'Hide captions' : 'Show captions'}
        >
          CC
        </button>

        <button
          className="control-btn"
          onClick={handleFullscreen}
          aria-label="Fullscreen"
          title="Fullscreen"
        >
          ⛶
        </button>
      </div>

      {/* Caption List (Clickable & Collapsible) */}
      {captions && captions.length > 0 && (
        <div className="caption-list-container">
          <div className="caption-list-header" onClick={() => setShowCaptionList(!showCaptionList)}>
            <h4>Captions (Click to seek)</h4>
            <button 
              className="collapse-btn"
              aria-label={showCaptionList ? 'Collapse captions' : 'Expand captions'}
              title={showCaptionList ? 'Collapse captions' : 'Expand captions'}
            >
              {showCaptionList ? '▼' : '▶'}
            </button>
          </div>
          {showCaptionList && (
            <div className="caption-list">
              {captions.map((caption, index) => (
                <div
                  key={index}
                  className={`caption-item ${
                    currentTime >= caption.start_time && currentTime <= caption.end_time
                      ? 'active'
                      : ''
                  }`}
                  onClick={() => handleCaptionClick(caption)}
                >
                  <span className="caption-time">
                    {formatTime(caption.start_time)} - {formatTime(caption.end_time)}
                  </span>
                  <span className="caption-text">{caption.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;

