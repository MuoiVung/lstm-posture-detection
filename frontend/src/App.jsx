import { useState, useEffect, useCallback } from "react";
import Header from "./components/Layout/Header";
import CameraFeed from "./components/Camera/CameraFeed";
import PostureStatus from "./components/Dashboard/PostureStatus";
import HealthRisks from "./components/Dashboard/HealthRisks";
import PostureHistory from "./components/Dashboard/PostureHistory";
import PostureAlert from "./components/Alerts/PostureAlert";
import TipsSection from "./components/Dashboard/TipsSection";
import CalibrationOverlay from "./components/Calibration/CalibrationOverlay";
import useWebSocket from "./hooks/useWebSocket";
import usePostureAnalysis from "./hooks/usePostureAnalysis";

/**
 * PostureGuard - Main Application
 *
 * App States:
 *   idle         → Camera off, nothing running
 *   preparing    → Camera on, 5s countdown for user to get in position
 *   calibrating  → Camera on, collecting baseline frames
 *   calibrated   → Calibration complete, ready to monitor
 *   monitoring   → Actively detecting posture
 */
export default function App() {
  const ws = useWebSocket();
  const analysis = usePostureAnalysis();

  // App state machine: idle → preparing → calibrating → calibrated → monitoring
  const [appState, setAppState] = useState("idle");
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [prepTime, setPrepTime] = useState(5);

  // Process incoming WebSocket messages
  useEffect(() => {
    const msg = ws.lastMessage;
    if (!msg) return;

    if (msg.type === "calibrating") {
      setCalibrationProgress(msg.calibration_progress || 0);
    } else if (msg.type === "calibration_complete") {
      setAppState("calibrated");
      setCalibrationProgress(1);
      // Auto-transition to monitoring after 1.5s
      setTimeout(() => {
        setAppState("monitoring");
      }, 1500);
    } else if (msg.type === "needs_calibration") {
      // Backend says we need to calibrate
      if (appState === "monitoring") {
        setAppState("idle");
      }
    } else if (msg.type === "prediction") {
      analysis.updateFromPrediction(msg);
    }
  }, [ws.lastMessage]);

  // Handle preparation countdown locally
  useEffect(() => {
    if (appState === "preparing") {
      if (prepTime > 0) {
        const timer = setTimeout(() => {
          setPrepTime(prev => prev - 1);
        }, 1000);
        return () => clearTimeout(timer);
      } else {
        // Prep time is over, move to actual calibration
        setAppState("calibrating");
        if (ws.status === "connected") {
          ws.sendCommand({ action: "start_calibration" });
        }
      }
    }
  }, [appState, prepTime, ws]);

  // When WS connects during calibrating state (if it was slow to connect), send the calibration command
  useEffect(() => {
    if (ws.status === "connected" && appState === "calibrating" && calibrationProgress === 0) {
      ws.sendCommand({ action: "start_calibration" });
    }
  }, [ws.status, appState, calibrationProgress]);

  // Start process: connect WebSocket + camera, start 5s preparation phase
  const handleStartCalibration = useCallback(() => {
    analysis.resetStats();
    setCalibrationProgress(0);
    setPrepTime(5);
    setAppState("preparing");

    if (ws.status !== "connected") {
      ws.connect();
    }
  }, [ws, analysis]);

  // Recalibrate: restart calibration flow with preparation phase
  const handleRecalibrate = useCallback(() => {
    setCalibrationProgress(0);
    setPrepTime(5);
    setAppState("preparing");
  }, []);

  // Stop everything
  const handleStop = useCallback(() => {
    ws.disconnect();
    setAppState("idle");
    setCalibrationProgress(0);
  }, [ws]);

  const isActive = appState === "preparing" || appState === "calibrating" || appState === "calibrated" || appState === "monitoring";
  const isMonitoring = appState === "monitoring";

  return (
    <div className="app-layout">
      <Header
        isActive={isMonitoring}
        connectionStatus={ws.status}
        appState={appState}
        onStartCalibration={handleStartCalibration}
        onRecalibrate={handleRecalibrate}
        onStop={handleStop}
      />

      {/* Floating Alerts */}
      <PostureAlert
        alerts={analysis.alerts}
        onDismiss={analysis.dismissAlert}
      />

      {/* Main Content Grid */}
      <main className="app-main">
        {/* Left Column: Camera */}
        <div className="camera-section">
          <CameraFeed
            isActive={isActive}
            sendFrame={ws.sendFrame}
            landmarks={ws.lastMessage?.landmarks}
            postureClass={
              analysis.currentPosture?.class || "good_posture"
            }
          >
            {/* Calibration Overlay (rendered inside camera-container) */}
            {(appState === "preparing" || appState === "calibrating" || appState === "calibrated") && (
              <CalibrationOverlay
                appState={appState}
                progress={calibrationProgress}
                prepTime={prepTime}
              />
            )}
          </CameraFeed>

          {/* Timeline below camera */}
          {isMonitoring && (
            <PostureHistory
              timeline={analysis.timeline}
              stats={analysis.stats}
            />
          )}
        </div>

        {/* Right Column: Dashboard */}
        <div className="right-panel">
          {isMonitoring ? (
            <>
              <PostureStatus posture={analysis.currentPosture} />
              <HealthRisks risks={analysis.healthRisks} />
            </>
          ) : (
            <div className="glass-card" style={{ textAlign: "center", padding: "3rem 2rem" }}>
              <div style={{ fontSize: "4rem", marginBottom: "1rem" }}>🦴</div>
              <h2 style={{ fontSize: "1.3rem", marginBottom: "0.5rem" }}>Welcome to PostureGuard</h2>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: "1.6", marginBottom: "1.5rem" }}>
                AI-powered posture monitoring that adapts to your setup.
                Click <strong>Start & Calibrate</strong> to begin.
              </p>
              <div style={{ 
                background: "var(--bg-glass)", 
                border: "1px solid var(--border-color)",
                borderRadius: "var(--radius-md)", 
                padding: "1rem", 
                textAlign: "left" 
              }}>
                <p style={{ fontSize: "0.8rem", fontWeight: "600", color: "var(--text-accent)", marginBottom: "0.5rem" }}>
                  📋 How it works:
                </p>
                <ol style={{ fontSize: "0.8rem", color: "var(--text-secondary)", paddingLeft: "1.2rem", lineHeight: "1.8" }}>
                  <li>Click <strong>"Start & Calibrate"</strong> in the header</li>
                  <li>Sit up straight with good posture for 5 seconds</li>
                  <li>The system learns YOUR baseline position</li>
                  <li>Monitoring begins automatically!</li>
                </ol>
              </div>
            </div>
          )}
          <TipsSection />
        </div>
      </main>
    </div>
  );
}
