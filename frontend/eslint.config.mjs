import coreWebVitals from "eslint-config-next/core-web-vitals";
import typescript from "eslint-config-next/typescript";

// eslint-config-next 16 ships native flat configs — spread them directly rather
// than wrapping in FlatCompat (which errors on the bundled react plugin).
const eslintConfig = [
  ...coreWebVitals,
  ...typescript,
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts"],
  },
  {
    rules: {
      // react-hooks v6 ships React-Compiler-adjacent rules that misfire on
      // idiomatic Next patterns. rules-of-hooks / exhaustive-deps stay as-is
      // (those catch real bugs — e.g. the hook-order defect we just fixed).
      //
      // error-boundaries flags every async server component's `try/catch →
      // fallback JSX` data-fetch pattern, which is the correct Next idiom — off.
      "react-hooks/error-boundaries": "off",
      // set-state-in-effect flags syncing local UI to external query state
      // (RefreshButton reacting to poll completion) — a legitimate use; keep it
      // visible as a warning rather than a blocking error.
      "react-hooks/set-state-in-effect": "warn",
    },
  },
];

export default eslintConfig;
