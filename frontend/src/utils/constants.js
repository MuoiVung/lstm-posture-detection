// API and WebSocket configuration constants

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

// Posture class metadata
export const POSTURE_CLASSES = {
  good_posture: {
    label: "Good Posture",
    emoji: "✅",
    color: "good",
    description: "Great job! Your posture is properly aligned.",
  },
  forward_lean: {
    label: "Forward Lean",
    emoji: "⚠️",
    color: "danger",
    description: "You're slouching forward. Sit up straight!",
  },
  backward_lean: {
    label: "Backward Lean",
    emoji: "⚠️",
    color: "warning",
    description: "You're leaning too far back. Adjust your position.",
  },
  left_lean: {
    label: "Left Lean",
    emoji: "↙️",
    color: "warning",
    description: "You're tilting to the left. Center yourself.",
  },
  right_lean: {
    label: "Right Lean",
    emoji: "↘️",
    color: "warning",
    description: "You're tilting to the right. Center yourself.",
  },
  head_forward: {
    label: "Head Forward",
    emoji: "🔴",
    color: "danger",
    description: "Your head is too far forward. Pull your chin back.",
  },
};

// MediaPipe pose connections for skeleton rendering
export const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
];

// Health tips
export const POSTURE_TIPS = [
  {
    title: "20-20-20 Rule",
    description: "Every 20 minutes, look at something 20 feet away for 20 seconds.",
    icon: "👁️",
  },
  {
    title: "Ergonomic Setup",
    description: "Keep your screen at eye level, arms at 90°, and feet flat.",
    icon: "🪑",
  },
  {
    title: "Regular Breaks",
    description: "Stand up and move for 2 minutes every 30 minutes.",
    icon: "🚶",
  },
  {
    title: "Core Strength",
    description: "Engage your core slightly while sitting for spinal support.",
    icon: "💪",
  },
  {
    title: "Shoulder Rolls",
    description: "Roll shoulders backward 10 times every hour.",
    icon: "🔄",
  },
  {
    title: "Chin Tucks",
    description: "Gently pull your chin back to correct forward head posture.",
    icon: "🧘",
  },
];

// Frame capture settings
export const CAPTURE_INTERVAL_MS = 100; // ~10fps to server
export const JPEG_QUALITY = 0.6;        // Balance quality vs bandwidth
