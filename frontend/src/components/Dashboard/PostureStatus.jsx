/**
 * PostureStatus component - displays current posture with animated confidence ring.
 */
export default function PostureStatus({ posture, onCalibrate }) {
  if (!posture) {
    return (
      <div className="glass-card posture-status" id="posture-status">
        <div className="empty-state">
          <span className="empty-state__icon">🦴</span>
          <p className="empty-state__text">
            Start monitoring to see your posture analysis in real-time
          </p>
        </div>
      </div>
    );
  }

  const { meta, confidence, isGood, probabilities } = posture;
  const percent = Math.round(confidence * 100);
  const statusClass = isGood ? "good" : percent > 70 ? "danger" : "warning";

  // SVG ring calculation
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - confidence);

  const ringColor = isGood ? "#00d4aa" : "#ef4444";

  // Sort probabilities for display
  const sortedProbs = Object.entries(probabilities || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div
      className={`glass-card glass-card--no-hover posture-status posture-status--${statusClass}`}
      id="posture-status"
    >
      {/* Confidence Ring */}
      <div className="posture-status__ring">
        <svg className="posture-status__ring-svg" viewBox="0 0 160 160">
          <circle
            className="posture-status__ring-bg"
            cx="80"
            cy="80"
            r={radius}
          />
          <circle
            className="posture-status__ring-progress"
            cx="80"
            cy="80"
            r={radius}
            stroke={ringColor}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
          />
        </svg>
        <div className="posture-status__ring-label">
          <span className="posture-status__emoji">{meta?.emoji || "🦴"}</span>
          <span className="posture-status__confidence">{percent}%</span>
        </div>
      </div>

      {/* Status Text */}
      <h2 className="posture-status__class-name">
        {meta?.label || posture.class}
      </h2>
      <p className="posture-status__description">
        {meta?.description || "Analyzing your posture..."}
      </p>

      {/* Calibration Button */}
      {posture.class === "needs_calibration" && (
        <button 
          className="button button--primary" 
          onClick={onCalibrate}
          style={{ width: '100%', marginTop: '1rem' }}
        >
          <span className="button__icon">🎯</span>
          Calibrate Now
        </button>
      )}

      {/* Recalibrate Button (Always available if already calibrated and not in needs_calibration) */}
      {posture.class !== "needs_calibration" && (
        <button 
          className="button button--outline" 
          onClick={onCalibrate}
          style={{ width: '100%', marginTop: '1rem', padding: '0.5rem', fontSize: '0.9rem' }}
        >
          <span className="button__icon">🔄</span>
          Recalibrate
        </button>
      )}

      {/* Probability Bars */}
      {sortedProbs.length > 0 && (
        <div className="probability-bars">
          {sortedProbs.map(([cls, prob]) => (
            <div key={cls} className="probability-bar">
              <span className="probability-bar__label">
                {cls.replace(/_/g, " ")}
              </span>
              <div className="probability-bar__track">
                <div
                  className="probability-bar__fill"
                  style={{ width: `${Math.round(prob * 100)}%` }}
                />
              </div>
              <span className="probability-bar__value">
                {Math.round(prob * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
