const STATUSES = {
  healthy: { label: "Healthy", state: "good", order: 3 },
  stale: { label: "Stale", state: "warn", order: 1 },
  degraded: { label: "Degraded", state: "warn", order: 1 },
  down: { label: "Down", state: "crit", order: 0 },
  unknown: { label: "Unknown", state: "neutral", order: 2 },
};

export function healthStatus(status) {
  return Object.hasOwn(STATUSES, status) ? STATUSES[status] : STATUSES.unknown;
}

const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

/** Keep the backend's source-specific calendar/cadence decisions authoritative. */
export function normalizeHealth(payload) {
  if (!record(payload) || !record(payload.sources)) throw new Error("Invalid source health response");
  const sources = Object.entries(payload.sources).map(([key, value]) => {
    if (!record(value)) throw new Error("Invalid source health entry");
    const status = typeof value.status === "string" ? value.status : "unknown";
    return {
      key,
      name: typeof value.name === "string" ? value.name : key,
      status,
      state: healthStatus(status).state,
      last_update: typeof value.last_update === "string" ? value.last_update : null,
      age_hours: typeof value.age_hours === "number" && Number.isFinite(value.age_hours) && value.age_hours >= 0 ? value.age_hours : null,
      message: typeof value.message === "string" && value.message ? value.message : "No reason reported",
    };
  }).sort((a, b) => healthStatus(a.status).order - healthStatus(b.status).order || a.name.localeCompare(b.name));
  return {
    overall_status: sources.length && typeof payload.overall_status === "string" ? payload.overall_status : "unknown",
    timestamp: typeof payload.timestamp === "string" ? payload.timestamp : null,
    total_sources: sources.length,
    sources,
  };
}

export function filterSources(sources, status, query) {
  const term = query.trim().toLowerCase();
  return sources.filter((source) =>
    (status === "all" || (status === "attention" ? source.status !== "healthy" : status === "unknown" ? healthStatus(source.status).label === "Unknown" : source.status === status)) &&
    [source.name, source.key, source.message].some((value) => value.toLowerCase().includes(term)));
}

/** Do not reinterpret timezone-less backend dates as the browser's local time. */
export function formatHealthDate(value) {
  if (!value || !/^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$/.test(value)) return "Not reported";
  const day = value.slice(0, 10);
  if (!Number.isFinite(Date.parse(value)) || new Date(`${day}T00:00:00Z`).toISOString().slice(0, 10) !== day) return "Not reported";
  if (!value.includes("T")) return value;
  const formatted = value.replace("T", " ").replace(/\.\d+/, "");
  return /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? formatted : `${formatted} (backend time)`;
}

export function formatHealthAge(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)}h elapsed` : "Not reported";
}
