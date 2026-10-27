import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveRefreshCompletion } from "./refreshOutcome.mjs";

describe("resolveRefreshCompletion", () => {
  it("treats a successful outcome as updated data", () => {
    assert.deepEqual(
      resolveRefreshCompletion({ outcome: "successful", failed_phases: [], error: null }),
      {
        outcome: "successful",
        message: "Data updated",
        shouldRefresh: true,
      },
    );
  });

  it("reports every failed phase for a partial outcome", () => {
    assert.deepEqual(
      resolveRefreshCompletion({
        outcome: "partial",
        failed_phases: ["FRED data", "market breadth"],
        error: null,
      }),
      {
        outcome: "partial",
        message: "Refresh partially completed — failed phases: FRED data, market breadth",
        shouldRefresh: true,
      },
    );
  });

  it("does not report updated data for a failed outcome", () => {
    assert.deepEqual(
      resolveRefreshCompletion({
        outcome: "failed",
        failed_phases: [],
        error: "database unavailable",
      }),
      {
        outcome: "failed",
        message: "Refresh failed: database unavailable",
        shouldRefresh: false,
      },
    );
  });

  it("treats a terminal status without a last run as unconfirmed", () => {
    assert.deepEqual(resolveRefreshCompletion(null), {
      outcome: "unconfirmed",
      message: "Couldn't confirm refresh — check data freshness",
      shouldRefresh: false,
    });
  });

  it("derives failure from error for older backend payloads", () => {
    assert.deepEqual(resolveRefreshCompletion({ failed_phases: [], error: "upstream timeout" }), {
      outcome: "failed",
      message: "Refresh failed: upstream timeout",
      shouldRefresh: false,
    });
  });

  it("derives failure when an older backend reports an empty error string", () => {
    assert.deepEqual(resolveRefreshCompletion({ failed_phases: [], error: "" }), {
      outcome: "failed",
      message: "Refresh failed",
      shouldRefresh: false,
    });
  });

  it("derives partial completion from failed phases for older backend payloads", () => {
    assert.deepEqual(resolveRefreshCompletion({ failed_phases: ["Yahoo prices"], error: null }), {
      outcome: "partial",
      message: "Refresh partially completed — failed phases: Yahoo prices",
      shouldRefresh: true,
    });
  });

  it("derives success for an older backend payload without errors or failed phases", () => {
    assert.deepEqual(resolveRefreshCompletion({ failed_phases: [], error: null }), {
      outcome: "successful",
      message: "Data updated",
      shouldRefresh: true,
    });
  });
});
