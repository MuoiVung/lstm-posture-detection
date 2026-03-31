"""
Health risk API router.
"""

from fastapi import APIRouter
from typing import List

from app.models.schemas import HealthRisk
from app.services.health_service import HEALTH_RISK_MAP

router = APIRouter()


@router.get("/risks/{posture_class}", response_model=List[HealthRisk])
async def get_risks_for_posture(posture_class: str):
    """Get health risks for a specific posture class."""
    risks = HEALTH_RISK_MAP.get(posture_class, [])
    return risks


@router.get("/risks", response_model=dict)
async def get_all_risks():
    """Get all health risk mappings."""
    return {
        posture: [r.model_dump() for r in risks]
        for posture, risks in HEALTH_RISK_MAP.items()
    }


@router.get("/tips")
async def get_posture_tips():
    """Get general posture improvement tips."""
    return {
        "tips": [
            {
                "title": "20-20-20 Rule",
                "description": "Every 20 minutes, look at something 20 feet away for 20 seconds.",
                "icon": "👁️",
            },
            {
                "title": "Ergonomic Setup",
                "description": "Keep your screen at eye level, arms at 90°, and feet flat on the floor.",
                "icon": "🪑",
            },
            {
                "title": "Regular Breaks",
                "description": "Stand up and move for 2 minutes every 30 minutes.",
                "icon": "🚶",
            },
            {
                "title": "Core Strengthening",
                "description": "Engage your core muscles slightly while sitting for better spinal support.",
                "icon": "💪",
            },
            {
                "title": "Shoulder Rolls",
                "description": "Roll your shoulders backward 10 times every hour to relieve tension.",
                "icon": "🔄",
            },
            {
                "title": "Chin Tucks",
                "description": "Gently pull your chin straight back to correct forward head posture.",
                "icon": "🧘",
            },
        ]
    }
