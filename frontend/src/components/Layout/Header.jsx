import { useState, useEffect } from "react";

/**
 * Header component with branding, session timer, and connection status.
 */
export default function Header({ isActive, connectionStatus, onToggle }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let interval;
    if (isActive) {
      interval = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      setElapsed(0);
    }
    return () => clearInterval(interval);
  }, [isActive]);

  const formatTime = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  const statusClass = {
    connected: "connection-badge--connected",
    connecting: "connection-badge--connecting",
    disconnected: "connection-badge--disconnected",
  }[connectionStatus] || "connection-badge--disconnected";

  const statusLabel = {
    connected: "Live",
    connecting: "Connecting...",
    disconnected: "Offline",
  }[connectionStatus] || "Offline";

  return (
    <header className="header" id="app-header">
      <div className="header__brand">
        <span className="header__logo">🦴</span>
        <div>
          <h1 className="header__title">PostureGuard</h1>
          <span className="header__subtitle">AI Posture Monitor</span>
        </div>
      </div>

      <div className="header__actions">
        <div className={`connection-badge ${statusClass}`}>
          <span
            className={`header__status-dot ${
              connectionStatus === "connected"
                ? "header__status-dot--active"
                : "header__status-dot--inactive"
            }`}
          />
          {statusLabel}
        </div>

        {isActive && (
          <div className="header__session-timer" id="session-timer">
            {formatTime(elapsed)}
          </div>
        )}

        <button
          className={`btn ${isActive ? "btn--danger" : "btn--primary"}`}
          onClick={onToggle}
          id="toggle-monitoring-btn"
        >
          {isActive ? "⏹ Stop" : "▶ Start Monitoring"}
        </button>
      </div>
    </header>
  );
}
