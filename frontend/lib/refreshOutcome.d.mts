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

export function resolveRefreshCompletion(lastRun: RefreshRun | null): RefreshCompletion;
