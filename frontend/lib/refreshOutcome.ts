export type RefreshOutcome = "successful" | "partial" | "failed";

export type RefreshRun = {
  outcome?: RefreshOutcome;
  failed_phases?: string[];
  error?: string | null;
};

export type RefreshCompletion = {
  outcome: RefreshOutcome | "unconfirmed";
  message: string;
  shouldRefresh: boolean;
};

const failedPhaseMessage = (prefix: string, failedPhases: string[]) =>
  failedPhases.length > 0
    ? `${prefix} — failed phases: ${failedPhases.join(", ")}`
    : `${prefix} — failed phases were not reported`;

export function resolveRefreshCompletion(lastRun: RefreshRun | null): RefreshCompletion {
  if (!lastRun) {
    return {
      outcome: "unconfirmed",
      message: "Couldn't confirm refresh — check data freshness",
      shouldRefresh: false,
    };
  }

  const failedPhases = lastRun.failed_phases ?? [];
  const hasError = lastRun.error !== null && lastRun.error !== undefined;
  const outcome =
    lastRun.outcome ?? (hasError ? "failed" : failedPhases.length > 0 ? "partial" : "successful");

  if (outcome === "successful") {
    return { outcome, message: "Data updated", shouldRefresh: true };
  }

  if (outcome === "partial") {
    return {
      outcome,
      message: failedPhaseMessage("Refresh partially completed", failedPhases),
      shouldRefresh: true,
    };
  }

  return {
    outcome,
    message: lastRun.error
      ? `Refresh failed: ${lastRun.error}`
      : failedPhases.length > 0
        ? failedPhaseMessage("Refresh failed", failedPhases)
        : "Refresh failed",
    shouldRefresh: false,
  };
}
