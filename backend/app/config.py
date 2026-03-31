"""
Application configuration loaded from environment variables.
"""

import os


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/posture.db")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/ml/weights/posture_lstm.pth")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Model config (must match training config)
    INPUT_SIZE: int = 49         # 13 joints × 3 coords + 10 engineered
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 2
    NUM_CLASSES: int = 5
    BIDIRECTIONAL: bool = True

    # Processing config
    WINDOW_SIZE: int = 30        # Frames per LSTM sequence
    FRAME_SKIP: int = 2          # Process every Nth frame to reduce load
    CONFIDENCE_THRESHOLD: float = 0.6  # Min confidence for a valid prediction

    # Posture class names
    CLASS_NAMES = [
        "good_posture",
        "forward_lean",
        "backward_lean",
        "left_lean",
        "right_lean",
    ]


settings = Settings()
