import { API_URL } from "../utils/constants";

/**
 * API service for REST communication with the backend.
 */
const api = {
  /**
   * Create a new monitoring session.
   */
  async createSession(userName = "default") {
    const res = await fetch(`${API_URL}/api/sessions/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_name: userName }),
    });
    return res.json();
  },

  /**
   * End a monitoring session.
   */
  async endSession(sessionId) {
    const res = await fetch(`${API_URL}/api/sessions/${sessionId}/end`, {
      method: "PUT",
    });
    return res.json();
  },

  /**
   * List recent sessions.
   */
  async listSessions(limit = 20) {
    const res = await fetch(`${API_URL}/api/sessions/?limit=${limit}`);
    return res.json();
  },

  /**
   * Get health risks for a posture class.
   */
  async getHealthRisks(postureClass) {
    const res = await fetch(`${API_URL}/api/health/risks/${postureClass}`);
    return res.json();
  },

  /**
   * Get posture improvement tips.
   */
  async getTips() {
    const res = await fetch(`${API_URL}/api/health/tips`);
    return res.json();
  },

  /**
   * Check API status.
   */
  async getStatus() {
    const res = await fetch(`${API_URL}/api/status`);
    return res.json();
  },
};

export default api;
