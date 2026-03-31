# 🦴 PostureGuard - AI Sitting Posture Monitor

Real-time sitting posture detection and health risk prediction using webcam, MediaPipe pose estimation, and LSTM deep learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![React](https://img.shields.io/badge/React-19-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐     ┌──────────────┐
│   Frontend   │◄──────────────────►│   Backend    │     │   Training   │
│   (React)    │   frames/results   │   (FastAPI)  │     │  (PyTorch)   │
│              │                    │              │     │              │
│  • Webcam    │     REST API       │  • MediaPipe │     │  • LSTM      │
│  • Dashboard │◄──────────────────►│  • LSTM Inf. │◄────│  • Dataset   │
│  • Alerts    │   sessions/health  │  • Health    │     │  • Evaluate  │
│  • Skeleton  │                    │  • SQLite    │     │  • Collect   │
└─────────────┘                    └──────────────┘     └──────────────┘
```

## Features

- 📷 **Real-time webcam** pose detection via MediaPipe
- 🧠 **LSTM model** (PyTorch) classifies 6 sitting postures
- 🩺 **Health risk prediction** with severity levels
- 📊 **Live dashboard** with posture timeline and statistics
- 🔔 **Smart alerts** for prolonged bad posture
- 💡 **Posture improvement tips**
- 🐳 **Docker support** for easy deployment
- 🍎 **Cross-platform**: macOS (Apple Silicon) + Windows (NVIDIA/CPU)

## Posture Classes

| Class | Description | Health Impact |
|-------|-------------|---------------|
| ✅ Good Posture | Upright, aligned | None |
| ⚠️ Forward Lean | Slouching forward | Back pain, disc issues |
| ⚠️ Backward Lean | Reclining too much | Lumbar pressure |
| ↙️ Left Lean | Tilting left | Spinal asymmetry |
| ↘️ Right Lean | Tilting right | Spinal asymmetry |
| 🔴 Head Forward | Tech neck | Neck pain, headaches |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (For training) Python 3.11+, webcam

### 1. Run with Docker

```bash
# Clone the repo
git clone <repo-url>
cd posture-detection

# Start frontend + backend
docker-compose up --build

# Access the app
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### 2. Local Development

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## Training the Model

### Step 1: Prepare Dataset

```bash
cd training

# Option A: Generate sample data for testing
python data/download_dataset.py --dataset sample --output data/processed

# Option B: Download real datasets (see Dataset Guide below)
python data/download_dataset.py --dataset all  # Shows download instructions

# Option C: Collect your own data via webcam
python src/collect_data.py --output data/raw/self_collected
```

### Step 2: Preprocess (if using video/raw data)

```bash
python src/preprocess.py --input data/raw --output data/processed
```

### Step 3: Train

```bash
python src/train.py \
  --config config/train_config.yaml \
  --data data/processed \
  --output outputs
```

### Step 4: Evaluate

```bash
python src/evaluate.py \
  --model outputs/run_*/models/best_model.pth \
  --data data/processed
```

### Step 5: Deploy Model

```bash
cp outputs/run_*/models/best_model.pth backend/app/ml/weights/posture_lstm.pth
```

### Train with Docker

```bash
# Run training container
docker-compose --profile training run training \
  python src/train.py --config config/train_config.yaml

# Or generate sample data first
docker-compose --profile training run training \
  python data/download_dataset.py --dataset sample --output data/processed
```

## Dataset Guide

### Recommended Datasets

1. **MultiPosture Dataset (Zenodo)**
   - 4,800 frames, 11 body joints (MediaPipe), CSV format
   - Download Link: https://zenodo.org/records/10397500 (or search Zenodo for "MultiPosture")
   - Place files in: `training/data/raw/multiposture/`

*Note: While Kaggle has some posture image datasets, they usually contain raw images rather than pre-extracted MediaPipe landmarks. The easiest way to get an accurate model is to use the Zenodo dataset above or collect your own data.*

### Self-Collected Data

Use the built-in data collection tool:
```bash
python training/src/collect_data.py --camera 0 --output training/data/raw/self_collected
```

**Controls:**
- `1-6`: Select posture label
- `SPACE`: Start/stop recording
- `Q`: Save and quit
- `R`: Reset

## Project Structure

```
posture-detection/
├── docker-compose.yml          # Multi-service Docker setup
├── frontend/                   # React app (Vite)
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── hooks/              # WebSocket + analysis hooks
│   │   ├── services/           # API client
│   │   └── utils/              # Constants
│   └── Dockerfile
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── routers/            # WebSocket, sessions, health
│   │   ├── services/           # Pose, posture, health, features
│   │   ├── models/             # DB models, schemas
│   │   └── ml/                 # LSTM model + weights
│   └── Dockerfile
└── training/                   # PyTorch training module
    ├── src/                    # Model, dataset, train, evaluate
    ├── config/                 # Hyperparameters (YAML)
    └── data/                   # Datasets
```

## Device Support

The app auto-detects the best available computing device:

| Device | Platform | Priority |
|--------|----------|----------|
| CUDA | Windows/Linux with NVIDIA GPU | 1st |
| MPS | macOS with Apple Silicon (M1/M2/M3) | 2nd |
| CPU | Any platform | Fallback |

## API Documentation

When the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 19, Vite, Vanilla CSS |
| Backend | FastAPI, SQLAlchemy, SQLite |
| ML Model | PyTorch LSTM (Bidirectional) |
| Pose Detection | MediaPipe Pose |
| Camera | WebSocket + HTML5 Canvas |
| Deployment | Docker Compose |

## License

MIT
