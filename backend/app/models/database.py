"""
SQLite database models and initialization using SQLAlchemy async.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import os
import json

from app.config import settings

Base = declarative_base()


class Session(Base):
    """Monitoring session record."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(100), default="default")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=0)
    good_posture_frames = Column(Integer, default=0)
    alerts_count = Column(Integer, default=0)
    posture_summary = Column(Text, default="{}")  # JSON string

    @property
    def good_posture_percentage(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.good_posture_frames / self.total_frames) * 100

    @property
    def posture_summary_dict(self) -> dict:
        try:
            return json.loads(self.posture_summary)
        except (json.JSONDecodeError, TypeError):
            return {}

    def update_posture(self, posture_class: str):
        """Update session with a new posture frame."""
        self.total_frames += 1
        if posture_class == "good_posture":
            self.good_posture_frames += 1

        summary = self.posture_summary_dict
        summary[posture_class] = summary.get(posture_class, 0) + 1
        self.posture_summary = json.dumps(summary)


class PostureLog(Base):
    """Individual posture detection log entry."""
    __tablename__ = "posture_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    posture_class = Column(String(50))
    confidence = Column(Float)
    health_risks = Column(Text, default="[]")  # JSON string


# Async engine and session
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create all database tables."""
    # Ensure data directory exists
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Get async database session."""
    async with async_session() as session:
        yield session
