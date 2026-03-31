import { useEffect, useCallback } from "react";
import Header from "./components/Layout/Header";
import CameraFeed from "./components/Camera/CameraFeed";
import PostureStatus from "./components/Dashboard/PostureStatus";
import HealthRisks from "./components/Dashboard/HealthRisks";
import PostureHistory from "./components/Dashboard/PostureHistory";
import PostureAlert from "./components/Alerts/PostureAlert";
import TipsSection from "./components/Dashboard/TipsSection";
import useWebSocket from "./hooks/useWebSocket";
import usePostureAnalysis from "./hooks/usePostureAnalysis";

/**
 * PostureGuard - Main Application
 *
 * Real-time sitting posture detection and health risk prediction
 * using webcam, MediaPipe, and LSTM deep learning.
 */
export default function App() {
  const ws = useWebSocket();
  const analysis = usePostureAnalysis();

  const isActive = ws.status === "connected";

  // Process incoming WebSocket messages
  useEffect(() => {
    if (ws.lastMessage?.type === "prediction") {
      analysis.updateFromPrediction(ws.lastMessage);
    }
  }, [ws.lastMessage]);

  // Toggle monitoring on/off
  const handleToggle = useCallback(() => {
    if (isActive) {
      ws.disconnect();
      // Don't reset stats so user can review their session
    } else {
      analysis.resetStats();
      ws.connect();
    }
  }, [isActive, ws, analysis]);

  const handleCalibrate = useCallback(() => {
    ws.sendCommand({ action: "calibrate" });
  }, [ws]);

  return (
    <div className="app-layout">
      <Header
        isActive={isActive}
        connectionStatus={ws.status}
        onToggle={handleToggle}
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
          />

          {/* Timeline below camera */}
          <PostureHistory
            timeline={analysis.timeline}
            stats={analysis.stats}
          />
        </div>

        {/* Right Column: Dashboard */}
        <div className="right-panel">
          <PostureStatus 
            posture={analysis.currentPosture} 
            onCalibrate={handleCalibrate}
          />
          <HealthRisks risks={analysis.healthRisks} />
          <TipsSection />
        </div>
      </main>
    </div>
  );
}
