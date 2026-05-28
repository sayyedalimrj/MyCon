// Live event stream for one run.
//
// Connects to /api/runs/{id}/events/stream, replays the buffered events
// the broker sends first, then forwards live events. Auto-reconnects
// once on transient failures so a brief network blip doesn't drop the
// view, but stops cleanly once the run reaches a terminal kind.
//
// Returns the rolling event list plus connection state. We cap retained
// events at MAX_EVENTS to avoid runaway memory if a stage emits very
// noisy stdout.

import { useEffect, useReducer, useRef } from "react";
import { endpoints } from "../api/endpoints";
import type { RunEvent } from "../api/types";

const MAX_EVENTS = 5_000;

type ConnState = "idle" | "connecting" | "open" | "closed" | "error";

interface State {
  events: RunEvent[];
  state: ConnState;
  error: string | null;
}

type Action =
  | { type: "connecting" }
  | { type: "open" }
  | { type: "event"; event: RunEvent }
  | { type: "error"; message: string }
  | { type: "closed" }
  | { type: "reset" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "connecting":
      return { ...state, state: "connecting", error: null };
    case "open":
      return { ...state, state: "open", error: null };
    case "event": {
      const events = state.events.length >= MAX_EVENTS
        ? [...state.events.slice(-MAX_EVENTS + 1), action.event]
        : [...state.events, action.event];
      return { ...state, events };
    }
    case "error":
      return { ...state, state: "error", error: action.message };
    case "closed":
      return { ...state, state: "closed" };
    case "reset":
      return { events: [], state: "idle", error: null };
  }
}

const TERMINAL_KINDS = new Set([
  "run.finished",
  "run.failed",
  "run.cancelled",
]);

export function useRunStream(runId: string | null | undefined): State {
  const [state, dispatch] = useReducer(reducer, {
    events: [],
    state: "idle",
    error: null,
  });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;
    dispatch({ type: "reset" });
    let cancelled = false;
    let reconnectAttempts = 0;

    const connect = () => {
      if (cancelled) return;
      const url = endpoints.websocketUrl(runId);
      dispatch({ type: "connecting" });
      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        dispatch({ type: "error", message: (err as Error).message });
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => dispatch({ type: "open" });
      ws.onmessage = (msg) => {
        try {
          const ev = JSON.parse(String(msg.data)) as RunEvent;
          if (!ev || typeof ev !== "object" || !("kind" in ev)) {
            return;
          }
          dispatch({ type: "event", event: ev });
          if (TERMINAL_KINDS.has(ev.kind)) {
            // Server closes after a terminal event; mirror locally.
            ws.close();
          }
        } catch {
          /* malformed frame; ignore */
        }
      };
      ws.onerror = () => {
        // The browser does not give us details on errors; if we never got
        // an "open", we'll attempt one reconnect to differentiate
        // "backend not ready yet" from "permanent failure".
        dispatch({ type: "error", message: "websocket error" });
      };
      ws.onclose = (ev) => {
        if (cancelled) {
          dispatch({ type: "closed" });
          return;
        }
        // Clean close (1000) or terminal-kind close: stop reconnecting.
        if (ev.code === 1000 || reconnectAttempts >= 2) {
          dispatch({ type: "closed" });
          return;
        }
        reconnectAttempts += 1;
        // Brief backoff so a tight reconnect loop does not hammer the API.
        setTimeout(connect, 600 * reconnectAttempts);
      };
    };

    connect();
    return () => {
      cancelled = true;
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        try {
          ws.close();
        } catch {
          /* already closing */
        }
      }
      wsRef.current = null;
    };
  }, [runId]);

  return state;
}
