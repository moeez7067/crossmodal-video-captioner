"""
FastAPI main application for Multimodal Video Captioning & Summarization System.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from src.api.routes import video, health
from src.utils.logger import get_logger
import config

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal Video Captioning & Summarization API",
    description="API for processing videos and generating captions, transcripts, and summaries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video.router, prefix="/api")
app.include_router(health.router, prefix="/api")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "status": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation error",
            "status": 422,
            "detail": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "status": 500,
            "detail": str(exc) if config.API_DEBUG else "An error occurred"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Multimodal Video Captioning & Summarization API")
    logger.info(f"API running on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"CORS origins: {config.CORS_ORIGINS}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multimodal Video Captioning & Summarization API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Multimodal Video Captioning & Summarization API",
        "endpoints": {
            "upload": "/api/video/upload",
            "process": "/api/video/process",
            "status": "/api/video/status/{job_id}",
            "results": "/api/video/results/{job_id}",
            "download": "/api/video/download/{job_id}/{format}",
            "health": "/api/health"
        }
    }

