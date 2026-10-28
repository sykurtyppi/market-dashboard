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

const sectors = {
  as_of: "2026-10-27 16:05",
  sectors: [
    { ticker: "XLK", name: "Technology", category: "Cyclical", change_pct: 1.2, price: 180.5, state: "good" },
  ],
  rotation: { signal: "Risk-On", state: "good", interpretation: "Cyclicals leading", leading_sectors: ["Technology (+1.2%)"] },
  vix_term: [
    { maturity: "9D", days: 9, value: 16.2, symbol: "VIX9D", change: null, change_pct: null, estimated: false },
    { maturity: "1M", days: 30, value: 17.45, symbol: "VIX", change: 1.25, change_pct: 7.72, estimated: false },
    { maturity: "3M", days: 90, value: 19.1, symbol: "VIX3M", change: 1.65, change_pct: 9.46, estimated: false },
    { maturity: "6M", days: 180, value: 20.4, symbol: "VIX6M", change: 1.3, change_pct: 6.81, estimated: false },
    { maturity: "1Y", days: 365, value: 21.15, symbol: "VIX1Y", change: 0.75, change_pct: 3.68, estimated: false },
  ],
  vix_structure: "Contango",
  vix_source: "Yahoo Finance · Cboe volatility index closes",
  warnings: [],
};

let sectorsMode = "full";
const partialSectors = {
  ...sectors,
  as_of: "2026-10-28 12:00",
  vix_term: sectors.vix_term.filter((point) => point.symbol !== "VIX3M").map((point) =>
    point.symbol === "VIX6M" ? { ...point, change: null, change_pct: null } : point
  ),
  warnings: ["VIX3M unavailable; adjacent curve comparison suppressed."],
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
  if (request.url === "/api/sectors") {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify(sectorsMode === "partial" ? partialSectors : sectors));
    return;
  }
  if (request.url === "/__test/sectors-partial") {
    sectorsMode = "partial";
    response.writeHead(204);
    response.end();
    return;
  }
  if (request.url === "/__test/sectors-full") {
    sectorsMode = "full";
    response.writeHead(204);
    response.end();
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