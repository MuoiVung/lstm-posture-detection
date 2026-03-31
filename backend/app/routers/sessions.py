"""
Session management REST API router.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.models.database import get_db, Session as SessionModel
from app.models.schemas import SessionCreate, SessionResponse

router = APIRouter()


@router.post("/", response_model=SessionResponse)
async def create_session(
    data: SessionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Start a new monitoring session."""
    session = SessionModel(
        user_name=data.user_name or "default",
        started_at=datetime.utcnow(),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return SessionResponse(
        id=session.id,
        user_name=session.user_name,
        started_at=session.started_at.isoformat(),
        total_frames=0,
        good_posture_percentage=0.0,
        alerts_count=0,
    )


@router.get("/", response_model=list[SessionResponse])
async def list_sessions(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List recent monitoring sessions."""
    result = await db.execute(
        select(SessionModel).order_by(SessionModel.started_at.desc()).limit(limit)
    )
    sessions = result.scalars().all()

    return [
        SessionResponse(
            id=s.id,
            user_name=s.user_name,
            started_at=s.started_at.isoformat(),
            ended_at=s.ended_at.isoformat() if s.ended_at else None,
            total_frames=s.total_frames,
            good_posture_percentage=s.good_posture_percentage,
            alerts_count=s.alerts_count,
            posture_summary=s.posture_summary_dict,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific session by ID."""
    result = await db.execute(
        select(SessionModel).where(SessionModel.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session.id,
        user_name=session.user_name,
        started_at=session.started_at.isoformat(),
        ended_at=session.ended_at.isoformat() if session.ended_at else None,
        total_frames=session.total_frames,
        good_posture_percentage=session.good_posture_percentage,
        alerts_count=session.alerts_count,
        posture_summary=session.posture_summary_dict,
    )


@router.put("/{session_id}/end")
async def end_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """End a monitoring session."""
    result = await db.execute(
        select(SessionModel).where(SessionModel.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.ended_at = datetime.utcnow()
    await db.commit()

    return {"message": "Session ended", "session_id": session_id}
