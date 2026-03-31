"""
Health risk mapping service.

Maps detected posture classes and duration to predicted health risks
with severity levels and recommendations.
"""

from typing import List, Dict
from app.models.schemas import HealthRisk


# Health risk database keyed by posture class
HEALTH_RISK_MAP = {
    "forward_lean": [
        HealthRisk(
            name="Lower Back Pain",
            description="Slouching increases pressure on lumbar discs by up to 190%, leading to chronic lower back pain.",
            severity="high",
            body_part="back",
            recommendation="Sit upright with your back against the chair. Use lumbar support.",
        ),
        HealthRisk(
            name="Disc Herniation Risk",
            description="Prolonged forward lean can cause spinal disc compression and potential herniation.",
            severity="high",
            body_part="spine",
            recommendation="Take a break every 30 minutes. Stand up and stretch your spine.",
        ),
        HealthRisk(
            name="Shoulder Strain",
            description="Rounded shoulders from slouching cause rotator cuff strain and upper back tension.",
            severity="medium",
            body_part="shoulders",
            recommendation="Roll your shoulders back and squeeze your shoulder blades together.",
        ),
    ],
    "head_forward": [
        HealthRisk(
            name="Tech Neck Syndrome",
            description="Forward head posture adds 10 lbs of force per inch of forward tilt on cervical spine.",
            severity="high",
            body_part="neck",
            recommendation="Align your ears over your shoulders. Adjust screen to eye level.",
        ),
        HealthRisk(
            name="Chronic Headaches",
            description="Tension in suboccipital muscles from forward head posture triggers tension headaches.",
            severity="medium",
            body_part="neck",
            recommendation="Perform chin tucks: gently pull your chin straight back.",
        ),
        HealthRisk(
            name="Cervical Disc Degeneration",
            description="Sustained forward head position accelerates wear on cervical vertebrae.",
            severity="high",
            body_part="spine",
            recommendation="Take regular breaks to look away from the screen and stretch your neck.",
        ),
    ],
    "left_lean": [
        HealthRisk(
            name="Spinal Asymmetry",
            description="Consistent lateral leaning can lead to functional scoliosis and muscle imbalances.",
            severity="medium",
            body_part="spine",
            recommendation="Center yourself in your chair. Keep both feet flat on the floor.",
        ),
        HealthRisk(
            name="Hip Misalignment",
            description="Leaning to one side puts uneven pressure on hip joints.",
            severity="medium",
            body_part="hips",
            recommendation="Distribute your weight evenly on both sit bones.",
        ),
    ],
    "right_lean": [
        HealthRisk(
            name="Spinal Asymmetry",
            description="Consistent lateral leaning can lead to functional scoliosis and muscle imbalances.",
            severity="medium",
            body_part="spine",
            recommendation="Center yourself in your chair. Keep both feet flat on the floor.",
        ),
        HealthRisk(
            name="Hip Misalignment",
            description="Leaning to one side puts uneven pressure on hip joints.",
            severity="medium",
            body_part="hips",
            recommendation="Distribute your weight evenly on both sit bones.",
        ),
    ],
    "backward_lean": [
        HealthRisk(
            name="Lumbar Disc Pressure",
            description="Excessive backward lean increases pressure on the posterior lumbar discs.",
            severity="medium",
            body_part="back",
            recommendation="Maintain a slight natural curve in your lower back with proper lumbar support.",
        ),
        HealthRisk(
            name="Core Muscle Weakness",
            description="Reclining too much leaves core muscles disengaged, weakening them over time.",
            severity="low",
            body_part="back",
            recommendation="Engage your core slightly while sitting. Consider core-strengthening exercises.",
        ),
    ],
    "good_posture": [],  # No health risks for good posture
}

# Duration-based escalation thresholds (minutes)
DURATION_WARNINGS = {
    5: "You've maintained this posture for 5 minutes. Consider adjusting.",
    15: "15 minutes of poor posture. Take a short break and stretch.",
    30: "30 minutes of continuous poor posture! Increased risk of pain. Stand up and move.",
    60: "1 hour of poor posture detected. High risk of developing chronic issues.",
}


class HealthService:
    """Health risk prediction and tracking service."""

    def __init__(self):
        # Track duration of consecutive bad posture (in seconds)
        self.bad_posture_duration: float = 0.0
        self.last_posture: str = "good_posture"
        self.total_alerts: int = 0

    def get_health_risks(
        self, posture_class: str, duration_seconds: float = 0.0
    ) -> List[HealthRisk]:
        """Get health risks for a given posture class.

        Args:
            posture_class: Detected posture classification
            duration_seconds: How long this posture has been maintained

        Returns:
            List of HealthRisk objects
        """
        risks = list(HEALTH_RISK_MAP.get(posture_class, []))

        # Add duration-based warnings
        if posture_class != "good_posture":
            minutes = duration_seconds / 60
            for threshold_min, message in sorted(DURATION_WARNINGS.items()):
                if minutes >= threshold_min:
                    risks.append(
                        HealthRisk(
                            name=f"Prolonged Bad Posture ({int(minutes)} min)",
                            description=message,
                            severity="high" if minutes >= 30 else "medium",
                            body_part="general",
                            recommendation="Stand up, walk around, and stretch for at least 2 minutes.",
                        )
                    )

        return risks

    def update_tracking(self, posture_class: str, frame_interval: float = 0.1):
        """Update bad posture duration tracking.

        Args:
            posture_class: Current detected posture
            frame_interval: Time between frames in seconds

        Returns:
            Current bad posture duration in seconds
        """
        if posture_class != "good_posture":
            self.bad_posture_duration += frame_interval
        else:
            if self.bad_posture_duration > 0:
                self.total_alerts += 1
            self.bad_posture_duration = 0.0

        self.last_posture = posture_class
        return self.bad_posture_duration

    def get_summary(self) -> Dict:
        """Get tracking summary."""
        return {
            "bad_posture_duration": self.bad_posture_duration,
            "total_alerts": self.total_alerts,
            "last_posture": self.last_posture,
        }

    def reset(self):
        """Reset tracking state."""
        self.bad_posture_duration = 0.0
        self.total_alerts = 0
        self.last_posture = "good_posture"
