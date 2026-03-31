import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from PIL import Image, ImageTk

# Import posture math logic (same as backend)
from posture_math import get_posture_service

# Setup modern UI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

POSTURE_LABELS = {
    "good_posture": ("Good Posture ✅", "#00d4aa"),
    "forward_lean": ("Forward Lean ⚠️", "#ef4444"),
    "backward_lean": ("Backward Lean ⚠️", "#fbbf24"),
    "left_lean": ("Left Lean ↙️", "#fbbf24"),
    "right_lean": ("Right Lean ↘️", "#fbbf24"),
    "head_forward": ("Head Forward 🔴", "#ef4444"),
    "needs_calibration": ("Needs Calibration 🎯", "#8b5cf6"),
    "calibrating": ("Calibrating...", "#8b5cf6"),
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

class PostureGuardApp(ctk.CTk):
    def __init__(self, cap):
        super().__init__()

        self.title("PostureGuard Desktop")
        self.geometry("900x600")

        # Layout
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Panel (Camera)
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.camera_frame, text="Starting Camera...")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Right Panel (Dashboard)
        self.dash_frame = ctk.CTkFrame(self)
        self.dash_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        self.status_label = ctk.CTkLabel(self.dash_frame, text="PostureGuard", font=ctk.CTkFont(size=24, weight="bold"))
        self.status_label.pack(pady=20)

        self.desc_label = ctk.CTkLabel(self.dash_frame, text="Click Start & Calibrate", font=ctk.CTkFont(size=14))
        self.desc_label.pack(pady=5)

        self.start_btn = ctk.CTkButton(self.dash_frame, text="🎯 Start & Calibrate", command=self.start_calibration, height=40)
        self.start_btn.pack(pady=20, padx=20, fill="x")

        # State Variables
        self.cap = cap
        self.running = True
        self.state = "idle"  # idle, preparing, calibrating, monitoring
        self.prep_time = 5
        self.prep_start_time = 0

        self.posture_service = get_posture_service()

        # Start Camera Loop
        self.after(100, self.update_frame)

    def start_calibration(self):
        if self.state == "idle" or self.state == "monitoring":
            self.posture_service.reset_buffer()
            self.state = "preparing"
            self.prep_start_time = time.time()
            self.start_btn.configure(text="⏱️ Preparing...", state="disabled")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze Pose
            results = pose.process(rgb_frame)

            # Draw UI Overlays
            display_frame = self.process_app_state(frame, results)

            # Convert for Tkinter
            rgb_disp = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_disp)
            
            # Resize appropriately
            w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if w > 10 and h > 10:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")

        # 30 FPS Loop (~33ms)
        self.after(30, self.update_frame)

    def process_app_state(self, frame, results):
        if not results.pose_landmarks:
            return frame

        # Draw Skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Build feature dict for PostureService
        key_landmarks = {}
        for idx, name in enumerate(LANDMARK_NAMES):
            lm = results.pose_landmarks.landmark[idx]
            key_landmarks[name] = [lm.x, lm.y, lm.z]

        # Handle States
        if self.state == "preparing":
            elapsed = time.time() - self.prep_start_time
            left = max(0, 5 - int(elapsed))
            cv2.putText(frame, f"GET READY: {left}s", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            self.desc_label.configure(text=f"Sit straight and get ready! ({left}s)")
            
            if elapsed >= 5:
                self.state = "calibrating"
                self.posture_service.start_calibration()
                self.start_btn.configure(text="🎯 Calibrating...", state="disabled")

        elif self.state == "calibrating":
            prediction = self.posture_service.add_frame(key_landmarks)
            if prediction:
                if prediction.get("posture_class") == "calibration_complete":
                    self.state = "monitoring"
                    self.start_btn.configure(text="🔄 Recalibrate", state="normal")
                    self.desc_label.configure(text="Monitoring active!")
                    self.status_label.configure(text="Good Posture ✅", text_color="#00d4aa")
                else:
                    prog = prediction.get("calibration_progress", 0)
                    pct = int(prog * 100)
                    cv2.putText(frame, f"CALIBRATING: {pct}%", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    self.desc_label.configure(text=f"Hold still... {pct}%")

        elif self.state == "monitoring":
            prediction = self.posture_service.add_frame(key_landmarks)
            if prediction:
                p_class = prediction.get("posture_class", "good_posture")
                conf = prediction.get("confidence", 0)
                
                label, color = POSTURE_LABELS.get(p_class, ("Unknown", "#ffffff"))
                
                # Update Dashboard
                self.status_label.configure(text=f"{label} ({int(conf*100)}%)", text_color=color)
                
                if p_class != "good_posture":
                    cv2.putText(frame, "ADJUST POSTURE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        return frame

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    import sys
    print("Requesting Camera Access (Allow in macOS settings if prompted)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open Camera. Check macOS privacy settings.")
        sys.exit(1)
        
    app = PostureGuardApp(cap)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
