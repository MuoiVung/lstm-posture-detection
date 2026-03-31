import { useState, useRef, useCallback } from "react";
import { POSTURE_CLASSES } from "../utils/constants";

/**
 * Custom hook for tracking posture analysis state.
 *
 * Manages posture history, session statistics, and alert generation.
 */
export default function usePostureAnalysis() {
  const [currentPosture, setCurrentPosture] = useState(null);
  const [healthRisks, setHealthRisks] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [stats, setStats] = useState({
    totalFrames: 0,
    goodFrames: 0,
    alertsCount: 0,
    postureDistribution: {},
  });
  const [alerts, setAlerts] = useState([]);

  const consecutiveBadRef = useRef(0);
  const alertIdRef = useRef(0);

  const updateFromPrediction = useCallback((message) => {
    if (!message?.posture) return;

    const { posture, health_risks: risks } = message;
    const postureClass = posture.class;
    const isGood = posture.is_good;

    // Update current posture
    setCurrentPosture({
      class: postureClass,
      confidence: posture.confidence,
      probabilities: posture.probabilities || {},
      isGood,
      meta: POSTURE_CLASSES[postureClass] || POSTURE_CLASSES.good_posture,
    });

    // Update health risks
    if (risks?.length > 0) {
      setHealthRisks(risks);
    } else if (isGood) {
      setHealthRisks([]);
    }

    // Update timeline (keep last 200 entries)
    setTimeline((prev) => {
      const entry = {
        timestamp: Date.now(),
        class: postureClass,
        isGood,
        confidence: posture.confidence,
      };
      const next = [...prev, entry];
      return next.slice(-200);
    });

    // Update stats
    setStats((prev) => {
      const dist = { ...prev.postureDistribution };
      dist[postureClass] = (dist[postureClass] || 0) + 1;

      return {
        ...prev,
        totalFrames: prev.totalFrames + 1,
        goodFrames: prev.goodFrames + (isGood ? 1 : 0),
        postureDistribution: dist,
      };
    });

    // Track consecutive bad posture for alerts
    if (!isGood) {
      consecutiveBadRef.current += 1;

      // Generate alert at thresholds (30, 90, 180 consecutive bad frames)
      const badCount = consecutiveBadRef.current;
      if (badCount === 30 || badCount === 90 || badCount === 180) {
        const alert = {
          id: ++alertIdRef.current,
          type: badCount >= 180 ? "danger" : "warning",
          title:
            badCount >= 180
              ? "⚠️ Prolonged Bad Posture!"
              : "Posture Alert",
          message:
            POSTURE_CLASSES[postureClass]?.description ||
            "Please adjust your sitting position.",
          timestamp: Date.now(),
        };

        setAlerts((prev) => [...prev, alert].slice(-5));
        setStats((prev) => ({
          ...prev,
          alertsCount: prev.alertsCount + 1,
        }));
      }
    } else {
      consecutiveBadRef.current = 0;
    }
  }, []);

  const dismissAlert = useCallback((alertId) => {
    setAlerts((prev) => prev.filter((a) => a.id !== alertId));
  }, []);

  const resetStats = useCallback(() => {
    setStats({
      totalFrames: 0,
      goodFrames: 0,
      alertsCount: 0,
      postureDistribution: {},
    });
    setTimeline([]);
    setAlerts([]);
    setCurrentPosture(null);
    setHealthRisks([]);
    consecutiveBadRef.current = 0;
  }, []);

  const goodPercentage =
    stats.totalFrames > 0
      ? Math.round((stats.goodFrames / stats.totalFrames) * 100)
      : 0;

  return {
    currentPosture,
    healthRisks,
    timeline,
    stats: { ...stats, goodPercentage },
    alerts,
    updateFromPrediction,
    dismissAlert,
    resetStats,
  };
}
