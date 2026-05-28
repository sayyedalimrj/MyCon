// Tiny, dependency-free fetch wrapper.
//
// We avoid axios because:
// - the Phase 2 surface is small and entirely JSON;
// - shipping a smaller bundle keeps the GUI snappy on a research workstation;
// - a thin wrapper composes more naturally with TanStack Query.
//
// All endpoints share a single base URL (default "/api"). The only
// non-JSON case is the WebSocket events stream, which has its own helper.

import { ApiError } from "./types";

export interface RequestOptions {
  signal?: AbortSignal;
  headers?: Record<string, string>;
  body?: unknown;
}

export class ApiClient {
  constructor(private readonly baseUrl: string = "/api") {}

  private async request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
    const url = this.baseUrl + path;
    const init: RequestInit = {
      method,
      headers: {
        Accept: "application/json",
        ...(opts.body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...(opts.headers ?? {}),
      },
      signal: opts.signal,
    };
    if (opts.body !== undefined) {
      init.body = JSON.stringify(opts.body);
    }

    let response: Response;
    try {
      response = await fetch(url, init);
    } catch (err) {
      // Network-level failure (server down, CORS preflight fail, etc.).
      throw new ApiError(0, (err as Error).message || "network error", null);
    }

    if (!response.ok) {
      let body: unknown = null;
      try {
        body = await response.json();
      } catch {
        try {
          body = await response.text();
        } catch {
          /* ignore */
        }
      }
      const detail =
        (typeof body === "object" && body !== null && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `${response.status} ${response.statusText}`) || "request failed";
      throw new ApiError(response.status, detail, body);
    }

    if (response.status === 204) {
      return undefined as T;
    }
    return (await response.json()) as T;
  }

  get<T>(path: string, opts?: RequestOptions): Promise<T> {
    return this.request<T>("GET", path, opts);
  }

  post<T>(path: string, body?: unknown, opts?: RequestOptions): Promise<T> {
    return this.request<T>("POST", path, { ...opts, body });
  }

  /**
   * Build a fully-qualified WebSocket URL for the given API path.
   *
   * The backend mounts the events stream at `/api/runs/{id}/events/stream`;
   * we let the browser pick `ws:` vs `wss:` by mirroring the page's scheme.
   */
  websocketUrl(path: string): string {
    const proto = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss:" : "ws:";
    const host =
      typeof window !== "undefined" && window.location.host
        ? window.location.host
        : "127.0.0.1:5173";
    // baseUrl is "/api"; we prepend host and proto.
    return `${proto}//${host}${this.baseUrl}${path}`;
  }
}

export const defaultApiClient = new ApiClient();
