/**
 * PostureHistory component - visualizes posture quality over time.
 */
export default function PostureHistory({ timeline, stats }) {
  const goodPercentage = stats?.goodPercentage ?? 0;

  return (
    <div className="glass-card glass-card--no-hover" id="posture-history">
      {/* Timeline Chart */}
      <h3 className="posture-timeline__title">
        📊 Posture Timeline
      </h3>

      {timeline.length === 0 ? (
        <div className="empty-state" style={{ padding: "var(--space-lg) 0" }}>
          <span className="empty-state__icon">📈</span>
          <p className="empty-state__text">
            Posture data will appear here as you are monitored
          </p>
        </div>
      ) : (
        <div className="posture-timeline__chart">
          {timeline.map((entry, i) => {
            const height = Math.max(10, entry.confidence * 100);
            const colorClass = entry.isGood
              ? "posture-timeline__bar--good"
              : entry.confidence > 0.7
              ? "posture-timeline__bar--bad"
              : "posture-timeline__bar--warning";

            return (
              <div
                key={i}
                className={`posture-timeline__bar ${colorClass}`}
                style={{ height: `${height}%` }}
                title={`${entry.class.replace(/_/g, " ")} (${Math.round(
                  entry.confidence * 100
                )}%)`}
              />
            );
          })}
        </div>
      )}

      {/* Session Stats Grid */}
      <div className="session-stats" id="session-stats">
        <div className="stat-card">
          <div
            className={`stat-card__value ${
              goodPercentage >= 70
                ? "stat-card__value--good"
                : goodPercentage >= 40
                ? "stat-card__value--warning"
                : "stat-card__value--danger"
            }`}
          >
            {goodPercentage}%
          </div>
          <div className="stat-card__label">Good Posture</div>
        </div>

        <div className="stat-card">
          <div className="stat-card__value stat-card__value--info">
            {stats?.totalFrames || 0}
          </div>
          <div className="stat-card__label">Frames</div>
        </div>

        <div className="stat-card">
          <div
            className={`stat-card__value ${
              (stats?.alertsCount || 0) > 0
                ? "stat-card__value--warning"
                : "stat-card__value--good"
            }`}
          >
            {stats?.alertsCount || 0}
          </div>
          <div className="stat-card__label">Alerts</div>
        </div>

        <div className="stat-card">
          <div className="stat-card__value stat-card__value--good">
            {stats?.totalFrames
              ? Object.keys(stats.postureDistribution || {}).length
              : 0}
          </div>
          <div className="stat-card__label">Postures</div>
        </div>
      </div>
    </div>
  );
}
