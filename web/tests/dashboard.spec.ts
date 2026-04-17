import { expect, test } from "@playwright/test";

test.describe("dashboard smoke", () => {
  test("renders index and key panels", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/Digital Twin|fl-robots/i);
    await expect(page.locator("#scoreboard")).toBeVisible();
    await expect(page.locator(".capture-panel")).toBeVisible();
  });

  test("health endpoint returns ok", async ({ request }) => {
    const res = await request.get("/api/health");
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
  });

  test("ships baseline security headers", async ({ request }) => {
    const res = await request.get("/");
    const h = res.headers();
    expect(h["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(h["x-frame-options"]).toBe("DENY");
    expect(h["x-content-type-options"]).toBe("nosniff");
  });

  test("start/stop training via command API", async ({ request }) => {
    const start = await request.post("/api/command", { data: { command: "start_training" } });
    expect([200, 401]).toContain(start.status()); // 401 if FL_ROBOTS_API_TOKEN set
    if (start.status() === 200) {
      const stop = await request.post("/api/command", { data: { command: "stop_training" } });
      expect(stop.status()).toBe(200);
    }
  });

  test("rejects unknown commands with 400", async ({ request }) => {
    const bad = await request.post("/api/command", { data: { command: "nope" } });
    expect(bad.status()).toBe(400);
  });
});
