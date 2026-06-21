import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import './App.css';
import VideoUploader from './components/VideoUploader';
import ProcessingStatus from './components/ProcessingStatus';
import ResultsViewer from './components/ResultsViewer';
import HistorySidebar from './components/HistorySidebar';

const AppContent: React.FC = () => {
  const [historyOpen, setHistoryOpen] = useState<boolean>(false);
  const location = useLocation();
  
  // Extract jobId from current route
  const jobIdMatch = location.pathname.match(/\/results\/([^/]+)/);
  const currentJobId = jobIdMatch ? jobIdMatch[1] : undefined;

  return (
    <div className="App">
      <HistorySidebar 
        isOpen={historyOpen} 
        onClose={() => setHistoryOpen(false)}
        currentJobId={currentJobId}
      />
      <header className="App-header">
        <button 
          onClick={() => setHistoryOpen(true)} 
          className="history-menu-btn"
          title="View processing history"
          aria-label="Open history"
        >
          <span className="history-icon">☰</span>
        </button>
        <h1>Multimodal Video Captioning & Summarization</h1>
      </header>
      <main className="App-main">
        <Routes>
          <Route path="/" element={<VideoUploader />} />
          <Route path="/process/:jobId" element={<ProcessingStatus />} />
          <Route path="/results/:jobId" element={<ResultsViewer />} />
        </Routes>
      </main>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;

