import { useState, useRef, useCallback, useEffect } from "react";
import { WS_URL } from "../utils/constants";

/**
 * Custom hook for WebSocket connection to the pose endpoint.
 *
 * Returns connection state and methods to send/receive pose data.
 */
export default function useWebSocket() {
  const [status, setStatus] = useState("disconnected"); // disconnected | connecting | connected
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");

    const ws = new WebSocket(`${WS_URL}/ws/pose`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      console.log("WebSocket disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus("disconnected");
    };
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
    setLastMessage(null);
  }, []);

  const sendFrame = useCallback((blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(blob);
    }
  }, []);

  const sendCommand = useCallback((cmd) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    status,
    lastMessage,
    connect,
    disconnect,
    sendFrame,
    sendCommand,
  };
}
