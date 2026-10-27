import { describe, expect, it } from "vitest";

import { resolveRefreshCompletion } from "./refreshOutcome";

describe("resolveRefreshCompletion", () => {
  it("treats a successful outcome as updated data", () => {
    expect(resolveRefreshCompletion({ outcome: "successful", failed_phases: [], error: null })).toEqual({
      outcome: "successful",
      message: "Data updated",
      shouldRefresh: true,
    });
  });

  it("reports every failed phase for a partial outcome", () => {
    expect(
      resolveRefreshCompletion({
        outcome: "partial",
        failed_phases: ["FRED data", "market breadth"],
        error: null,
      }),
    ).toEqual({
      outcome: "partial",
      message: "Refresh partially completed — failed phases: FRED data, market breadth",
      shouldRefresh: true,
    });
  });

  it("does not report updated data for a failed outcome", () => {
    expect(
      resolveRefreshCompletion({ outcome: "failed", failed_phases: [], error: "database unavailable" }),
    ).toEqual({
      outcome: "failed",
      message: "Refresh failed: database unavailable",
      shouldRefresh: false,
    });
  });

  it("treats a terminal status without a last run as unconfirmed", () => {
    expect(resolveRefreshCompletion(null)).toEqual({
      outcome: "unconfirmed",
      message: "Couldn't confirm refresh — check data freshness",
      shouldRefresh: false,
    });
  });

  it("derives failure from error for older backend payloads", () => {
    expect(resolveRefreshCompletion({ failed_phases: [], error: "upstream timeout" })).toEqual({
      outcome: "failed",
      message: "Refresh failed: upstream timeout",
      shouldRefresh: false,
    });
  });

  it("derives failure when an older backend reports an empty error string", () => {
    expect(resolveRefreshCompletion({ failed_phases: [], error: "" })).toEqual({
      outcome: "failed",
      message: "Refresh failed",
      shouldRefresh: false,
    });
  });

  it("derives partial completion from failed phases for older backend payloads", () => {
    expect(resolveRefreshCompletion({ failed_phases: ["Yahoo prices"], error: null })).toEqual({
      outcome: "partial",
      message: "Refresh partially completed — failed phases: Yahoo prices",
      shouldRefresh: true,
    });
  });

  it("derives success for an older backend payload without errors or failed phases", () => {
    expect(resolveRefreshCompletion({ failed_phases: [], error: null })).toEqual({
      outcome: "successful",
      message: "Data updated",
      shouldRefresh: true,
    });
  });
});
