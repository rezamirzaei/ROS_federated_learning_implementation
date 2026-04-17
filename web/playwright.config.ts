import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for the fl-robots standalone dashboard.
 *
 * The Python server is launched as a background webServer; Playwright
 * waits for /api/health 200 before starting the suite.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: process.env.FL_ROBOTS_BASE_URL || "http://127.0.0.1:5055",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "python ../main.py run --host 127.0.0.1 --port 5055",
    url: "http://127.0.0.1:5055/api/health",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    cwd: ".",
  },
});
