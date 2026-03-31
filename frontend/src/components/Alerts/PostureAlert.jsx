import { useEffect } from "react";

/**
 * PostureAlert - floating notification alerts for posture warnings.
 */
export default function PostureAlert({ alerts, onDismiss }) {
  // Auto-dismiss after 5 seconds
  useEffect(() => {
    if (alerts.length === 0) return;

    const timers = alerts.map((alert) =>
      setTimeout(() => {
        onDismiss(alert.id);
      }, 5000)
    );

    return () => timers.forEach(clearTimeout);
  }, [alerts, onDismiss]);

  if (alerts.length === 0) return null;

  return (
    <div className="alert-container" id="alert-container">
      {alerts.map((alert) => (
        <div
          key={alert.id}
          className={`alert alert--${alert.type}`}
          onClick={() => onDismiss(alert.id)}
          role="alert"
        >
          <div className="alert__title">{alert.title}</div>
          <div className="alert__message">{alert.message}</div>
        </div>
      ))}
    </div>
  );
}
