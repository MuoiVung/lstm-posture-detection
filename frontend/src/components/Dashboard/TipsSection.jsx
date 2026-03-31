import { POSTURE_TIPS } from "../../utils/constants";

/**
 * Tips section showing posture improvement advice.
 */
export default function TipsSection() {
  return (
    <div className="glass-card glass-card--no-hover tips-section" id="tips-section">
      <h3 className="tips-section__title">💡 Posture Tips</h3>
      <div className="tips-grid">
        {POSTURE_TIPS.map((tip, i) => (
          <div key={i} className="tip-card">
            <div className="tip-card__icon">{tip.icon}</div>
            <div className="tip-card__title">{tip.title}</div>
            <div className="tip-card__description">{tip.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
