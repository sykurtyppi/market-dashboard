import { defineConfig } from "@playwright/test";

const frontendPort = Number(process.env.E2E_FRONTEND_PORT ?? 3100);
const backendPort = Number(process.env.E2E_BACKEND_PORT ?? 3101);

export default defineConfig({
  // Mock scenario controls are process-wide; serialize files and reset scenarios.
  workers: 1,
  testDir: "./tests/e2e",
  use: { baseURL: `http://127.0.0.1:${frontendPort}` },
  webServer: {
    command: `E2E_BACKEND_PORT=${backendPort} node tests/e2e/mock-backend.mjs & MOCK_PID=$!; trap 'kill $MOCK_PID' EXIT; MARKET_API_URL=http://127.0.0.1:${backendPort} npm run dev -- --hostname 127.0.0.1 --port ${frontendPort}`,
    url: `http://127.0.0.1:${frontendPort}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});