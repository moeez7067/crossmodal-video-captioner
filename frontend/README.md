# Multimodal Video Captioning - Frontend

React + TypeScript frontend application for the Multimodal Video Captioning & Summarization System.

## Features

- **Video Upload**: Drag-and-drop video file upload with validation
- **Processing Status**: Real-time progress tracking with status updates
- **Results Viewer**: View and download captions, transcripts, and summaries
- **Video Player**: Integrated video player with caption overlay support
- **Multiple Formats**: Support for SRT, VTT, TXT, DOCX, and PDF downloads

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- TypeScript (installed via npm)

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Configuration

Create a `.env` file in the frontend directory (optional):
```env
REACT_APP_API_URL=http://localhost:8000/api
```

If not set, the default API URL is `http://localhost:8000/api`.

## Running the Application

### Development Mode
```bash
npm start
```

The application will open at [http://localhost:3000](http://localhost:3000)

### Production Build
```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Project Structure

```
frontend/
├── public/              # Static files
├── src/
│   ├── components/     # React components (TypeScript)
│   │   ├── VideoUploader.tsx
│   │   ├── ProcessingStatus.tsx
│   │   ├── ResultsViewer.tsx
│   │   └── VideoPlayer.tsx
│   ├── services/       # API service layer
│   │   └── api.ts
│   ├── utils/          # Utility functions
│   │   └── formatters.ts
│   ├── App.tsx         # Main application component
│   ├── index.tsx        # Entry point
│   └── react-app-env.d.ts  # TypeScript declarations
├── tsconfig.json        # TypeScript configuration
├── package.json
└── README.md
```

## Components

### VideoUploader
Handles video file upload with drag-and-drop support, file validation, and upload progress tracking.

### ProcessingStatus
Displays real-time processing status with progress bar, current stage, and estimated time remaining.

### ResultsViewer
Shows processing results with tabs for captions, transcript, and summary. Includes format selection and download functionality.

### VideoPlayer
HTML5 video player with integrated caption overlay and playback controls.

## API Integration

The frontend communicates with the backend API through the `api.js` service layer. Make sure the backend API is running and accessible at the configured URL.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests

### Code Style

The project uses TypeScript with strict type checking. ESLint is configured with React app defaults. Code formatting follows standard React and TypeScript conventions.

## Troubleshooting

### API Connection Issues
- Verify the backend API is running
- Check the `REACT_APP_API_URL` environment variable
- Ensure CORS is properly configured on the backend

### Build Issues
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Clear npm cache: `npm cache clean --force`

## License

See main project LICENSE file.

