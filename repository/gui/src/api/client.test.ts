import { describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";

import { ApiClient } from "./client";
import { ApiError } from "./types";
import { server } from "../test/msw/server";

describe("ApiClient", () => {
  const client = new ApiClient("/api");

  it("parses JSON success", async () => {
    server.use(
      http.get("*/api/health", () => HttpResponse.json({ status: "ok", stage_count: 7 })),
    );
    const out = await client.get<{ status: string; stage_count: number }>("/health");
    expect(out).toEqual({ status: "ok", stage_count: 7 });
  });

  it("throws ApiError with detail on 4xx", async () => {
    server.use(
      http.get("*/api/configs/missing", () =>
        HttpResponse.json({ detail: "Unknown config: 'missing'" }, { status: 404 }),
      ),
    );
    await expect(client.get("/configs/missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
    });
    try {
      await client.get("/configs/missing");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).message).toMatch(/Unknown config/);
    }
  });

  it("posts JSON and parses the response", async () => {
    server.use(
      http.post("*/api/runs", async ({ request }) => {
        const body = (await request.json()) as { config_path: string };
        return HttpResponse.json({ run_id: "r-123", echoed: body.config_path }, { status: 201 });
      }),
    );
    const out = await client.post<{ run_id: string; echoed: string }>("/runs", {
      config_path: "/tmp/foo.yaml",
    });
    expect(out).toEqual({ run_id: "r-123", echoed: "/tmp/foo.yaml" });
  });

  it("throws ApiError with status 0 on network failure", async () => {
    server.use(http.get("*/api/health", () => HttpResponse.error()));
    await expect(client.get("/health")).rejects.toMatchObject({ status: 0 });
  });

  it("builds a websocket URL with the matching scheme", () => {
    // jsdom: window.location.protocol is "http:" by default so we expect "ws:".
    const url = client.websocketUrl("/runs/abc/events/stream");
    expect(url).toMatch(/^ws:\/\/[^/]+\/api\/runs\/abc\/events\/stream$/);
  });
});
