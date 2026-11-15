import test from "node:test";
import assert from "node:assert/strict";
import { normalizeHealth, healthStatus, filterSources, formatHealthDate, formatHealthAge } from "./sourceHealth.mjs";

const source = (status, extra = {}) => ({ name: "VIX", status, last_update: "2026-11-13T00:00:00", age_hours: 60, message: "Data current (last: 2026-11-13)", ...extra });

test("Friday data remains healthy on a weekend or holiday regardless of elapsed hours", () => {
  for (const age_hours of [60, 84]) {
    const health = normalizeHealth({ overall_status: "healthy", sources: { vix: source("healthy", { age_hours }) }, timestamp: "2026-11-15T12:00:00" });
    assert.equal(healthStatus(health.overall_status).label, "Healthy");
    assert.equal(health.sources[0].state, "good");
    assert.equal(health.sources[0].age_hours, age_hours);
  }
});

test("backend stale, degraded and down statuses are never hidden by a young snapshot", () => {
  for (const status of ["stale", "degraded", "down"]) {
    const health = normalizeHealth({ overall_status: status, sources: { vix: source(status, { age_hours: 0 }) } });
    assert.notEqual(healthStatus(health.overall_status).state, "good");
    assert.equal(health.sources[0].status, status);
  }
});

test("unknown, missing and unsupported health stay neutral instead of fresh", () => {
  assert.equal(healthStatus("new_status").state, "neutral");
  assert.equal(normalizeHealth({ sources: {} }).overall_status, "unknown");
  assert.equal(normalizeHealth({ sources: {} }).total_sources, 0);
  for (const bad of [null, [], { sources: [] }, { sources: { vix: null } }]) {
    assert.throws(() => normalizeHealth(bad), /Invalid source health/);
  }
});

test("sources are sorted by severity and filter by status, name, key or reason", () => {
  const health = normalizeHealth({ overall_status: "degraded", sources: {
    vix: source("healthy"),
    hy: source("stale", { name: "Credit", message: "Two business days behind" }),
    missing: source("unknown", { name: "RRP", last_update: null }),
    db: source("down", { name: "Database" }),
  } });
  assert.deepEqual(health.sources.map((s) => s.key), ["db", "hy", "missing", "vix"]);
  assert.equal(filterSources(health.sources, "attention", "").length, 3);
  assert.equal(filterSources(health.sources, "stale", " BUSINESS ")[0].key, "hy");
  assert.equal(filterSources(health.sources, "all", "vix")[0].key, "vix");
  assert.deepEqual(filterSources(health.sources, "healthy", "missing"), []);
});

test("inherited object keys are not mistaken for supported health statuses", () => {
  for (const status of ["constructor", "__proto__", "toString"]) {
    assert.equal(healthStatus(status).label, "Unknown");
    assert.equal(healthStatus(status).state, "neutral");
  }
});

test("invalid calendar dates are not displayed as reported backend observations", () => {
  for (const date of ["2026-99-99", "2026-02-30", "2026-11-15T99:99:99"]) {
    assert.equal(formatHealthDate(date), "Not reported");
  }
});

test("dates preserve backend date semantics without inventing timezone; missing values are explicit", () => {
  assert.equal(formatHealthDate("2026-11-13"), "2026-11-13");
  assert.equal(formatHealthDate("2026-11-15T12:05:14.211997"), "2026-11-15 12:05:14 (backend time)");
  assert.equal(formatHealthDate("2026-11-15T12:05:14Z"), "2026-11-15 12:05:14Z");
  assert.equal(formatHealthDate(null), "Not reported");
  assert.equal(formatHealthAge(null), "Not reported");
  assert.equal(formatHealthAge(NaN), "Not reported");
  assert.equal(formatHealthAge(60), "60.0h elapsed");
});
