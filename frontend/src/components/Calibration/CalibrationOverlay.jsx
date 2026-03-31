/**
 * CalibrationOverlay - Displays calibration progress or preparation countdown.
 */
export default function CalibrationOverlay({ appState, progress, prepTime }) {
  if (appState === "calibrated") {
    return (
      <div className="calibration-overlay calibration-overlay--complete">
        <div className="calibration-overlay__content">
          <div className="calibration-overlay__icon calibration-overlay__icon--success">
            ✅
          </div>
          <h3 className="calibration-overlay__title">Calibration Complete!</h3>
          <p className="calibration-overlay__subtitle">
            Starting posture monitoring...
          </p>
        </div>
      </div>
    );
  }

  if (appState === "preparing") {
    return (
      <div className="calibration-overlay">
        <div className="calibration-overlay__content">
          <div className="calibration-overlay__icon" style={{ animation: "none" }}>⏱️</div>
          <h3 className="calibration-overlay__title" style={{ color: "var(--color-warning)", background: "none", WebkitTextFillColor: "inherit" }}>
            Get Ready...
          </h3>
          <p className="calibration-overlay__instruction">
            Adjust your seat, sit up straight, and look at the screen.
          </p>

          <div className="calibration-overlay__progress-ring">
            <svg viewBox="0 0 120 120">
              <circle
                className="calibration-overlay__ring-bg"
                cx="60" cy="60" r="52"
              />
              <circle
                className="calibration-overlay__ring-fill"
                cx="60" cy="60" r="52"
                style={{ stroke: "var(--color-warning)" }}
                strokeDasharray={`${2 * Math.PI * 52}`}
                strokeDashoffset={`${2 * Math.PI * 52 * (1 - (prepTime / 5))}`}
              />
            </svg>
            <div className="calibration-overlay__progress-text">
              <span className="calibration-overlay__countdown" style={{ color: "var(--color-warning)" }}>
                {prepTime}
              </span>
              <span className="calibration-overlay__countdown-label">sec</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Calibrating state
  const percent = Math.round(progress * 100);
  // Backend takes ~2.5 seconds (25 frames at 10fps), so countdown from 3s is accurate
  const secondsLeft = Math.max(0, Math.ceil((1 - progress) * 2.5));

  return (
    <div className="calibration-overlay">
      <div className="calibration-overlay__content">
        <div className="calibration-overlay__icon">🎯</div>
        <h3 className="calibration-overlay__title">Calibrating...</h3>
        <p className="calibration-overlay__instruction">
          Hold your good posture perfectly still.
        </p>

        {/* Circular progress */}
        <div className="calibration-overlay__progress-ring">
          <svg viewBox="0 0 120 120">
            <circle
              className="calibration-overlay__ring-bg"
              cx="60" cy="60" r="52"
            />
            <circle
              className="calibration-overlay__ring-fill"
              cx="60" cy="60" r="52"
              strokeDasharray={`${2 * Math.PI * 52}`}
              strokeDashoffset={`${2 * Math.PI * 52 * (1 - progress)}`}
            />
          </svg>
          <div className="calibration-overlay__progress-text">
            <span className="calibration-overlay__countdown">
              {secondsLeft === 0 ? "..." : secondsLeft}
            </span>
            <span className="calibration-overlay__countdown-label">sec</span>
          </div>
        </div>

        <div className="calibration-overlay__percent">{percent}%</div>
      </div>
    </div>
  );
}
