import { createServer } from "node:http";

const overview = {
  as_of: "2026-10-27",
  freshness: { status: "fresh", age: "0h", is_fresh: true },
  warnings: [],
  left_signal: null,
  regime: { composite_risk: 42, breakdown: null, components: [] },
  metrics: [],
  charts: { vrp_history: [], credit_spreads: { hy: [], ig: [] } },
  detail: [],
};

const repo = {
  as_of: "2026-10-27",
  warnings: [],
  regime: "NORMAL",
  state: "good",
  regime_note: "Ample reserves — no funding stress.",
  metrics: [
    {
      key: "rrp",
      label: "RRP (Overnight)",
      value: 82.5,
      unit: "B",
      state: "neutral",
      source: "FRED (RRPONTSYD), $B",
    },
  ],
  charts: {
    sofr_history: [],
    rrp_history: [{ date: "2026-10-27", value: 82.5 }],
  },
};

const freshness = {
  status: "fresh",
  age: "0h",
  is_fresh: true,
  as_of: "2026-10-27",
  age_hours: 0,
};

createServer((request, response) => {
  if (request.url === "/api/overview") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(overview));
    return;
  }
  if (request.url === "/api/repo") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(repo));
    return;
  }
  if (request.url === "/api/freshness") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(freshness));
    return;
  }
  response.writeHead(404, { "content-type": "application/json" });
  response.end(JSON.stringify({ detail: "not found" }));
}).listen(3101, "127.0.0.1");