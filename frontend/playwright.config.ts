import { defineConfig } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_PORT ?? 3100);
const apiPort = Number(process.env.PLAYWRIGHT_API_PORT ?? 3101);

export default defineConfig({
  testDir: "./tests/e2e",
  // Fault controls mutate one shared mock API; do not leak them across test files.
  workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}` },
  webServer: {
    command: `PLAYWRIGHT_API_PORT=${apiPort} node tests/e2e/mock-backend.mjs & MOCK_PID=$!; trap 'kill $MOCK_PID' EXIT; MARKET_API_URL=http://127.0.0.1:${apiPort} npm run dev -- --hostname 127.0.0.1 --port ${port}`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});