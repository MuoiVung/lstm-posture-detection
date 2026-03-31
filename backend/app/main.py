"""
FastAPI application entry point for PostureGuard backend.

Provides:
- WebSocket endpoint for real-time pose streaming
- REST API for session management and health insights
- CORS middleware for frontend communication
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.config import settings
from app.routers import websocket, sessions, health
from app.models.database import init_db

app = FastAPI(
    title="PostureGuard API",
    description="Real-time sitting posture detection and health risk prediction",
    version="1.0.0",
)

# CORS middleware
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(websocket.router, tags=["WebSocket"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])


@app.on_event("startup")
async def startup():
    """Initialize database and load ML model on startup."""
    await init_db()
    print("✅ Database initialized")
    print(f"✅ PostureGuard API running")
    print(f"   CORS origins: {origins}")
    print(f"   Model path: {settings.MODEL_PATH}")


@app.get("/")
async def root():
    return {
        "name": "PostureGuard API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/status")
async def api_status():
    """Health check endpoint."""
    model_exists = os.path.exists(settings.MODEL_PATH)
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "device": settings.DEVICE,
    }
