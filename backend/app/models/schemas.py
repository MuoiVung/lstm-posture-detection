"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


# ─── Pose & Posture ────────────────────────────────────────

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class PoseResult(BaseModel):
    """Result from a single frame analysis."""
    posture_class: str
    confidence: float
    landmarks: Optional[List[LandmarkPoint]] = None
    health_risks: List["HealthRisk"] = []
    timestamp: float


class HealthRisk(BaseModel):
    """A predicted health risk based on posture."""
    name: str
    description: str
    severity: str          # "low", "medium", "high"
    body_part: str         # "neck", "back", "spine", "shoulders", "hips"
    recommendation: str


# ─── Session ───────────────────────────────────────────────

class SessionCreate(BaseModel):
    """Start a new monitoring session."""
    user_name: Optional[str] = "default"


class SessionResponse(BaseModel):
    id: int
    user_name: str
    started_at: str
    ended_at: Optional[str] = None
    total_frames: int
    good_posture_percentage: float
    alerts_count: int
    posture_summary: Optional[Dict[str, int]] = None


class SessionUpdate(BaseModel):
    """Update session with new posture data."""
    posture_class: str
    confidence: float


# ─── Stats ─────────────────────────────────────────────────

class PostureStats(BaseModel):
    """Aggregated posture statistics."""
    total_frames: int
    good_posture_percentage: float
    session_duration_minutes: float
    posture_distribution: Dict[str, int]
    current_streak_good: int
    alerts: List[str]


class DashboardData(BaseModel):
    """Full dashboard data payload."""
    current_posture: Optional[PoseResult] = None
    session_stats: Optional[PostureStats] = None
    health_risks: List[HealthRisk] = []
    posture_timeline: List[Dict] = []


# Fix forward reference
PoseResult.model_rebuild()
