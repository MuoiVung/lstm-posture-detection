/**
 * HealthRisks component - displays health risk prediction cards.
 */
export default function HealthRisks({ risks }) {
  if (!risks || risks.length === 0) {
    return (
      <div className="glass-card glass-card--no-hover" id="health-risks">
        <h3 className="health-risks__title">🩺 Health Risks</h3>
        <div className="no-risks">
          <span className="no-risks__icon">✅</span>
          <div>
            <p className="no-risks__text">No health risks detected</p>
            <p className="no-risks__subtext">
              Keep maintaining good posture!
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--no-hover" id="health-risks">
      <h3 className="health-risks__title">
        🩺 Health Risks
        <span
          style={{
            fontSize: "0.75rem",
            color: "var(--text-muted)",
            fontWeight: 400,
          }}
        >
          ({risks.length} detected)
        </span>
      </h3>
      <div className="health-risks">
        {risks.map((risk, i) => (
          <div
            key={`${risk.name}-${i}`}
            className={`health-risk-card health-risk-card--${risk.severity}`}
          >
            <div className="health-risk-card__header">
              <span className="health-risk-card__name">{risk.name}</span>
              <span
                className={`health-risk-card__severity health-risk-card__severity--${risk.severity}`}
              >
                {risk.severity}
              </span>
            </div>
            <p className="health-risk-card__description">{risk.description}</p>
            <div className="health-risk-card__recommendation">
              <span>💡</span>
              <span>{risk.recommendation}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
