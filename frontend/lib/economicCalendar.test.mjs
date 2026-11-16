// Permanent regression: exercise the actual economic-calendar page against a
// controlled API + JSX harness and assert the GDP level/Prior cells, zero,
// missing, separate YoY, FOMC two-decimal rate, and CPI plain level.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const ts = require("typescript");

const page = readFileSync(
  new URL("../app/economic-calendar/page.tsx", import.meta.url),
  "utf8",
);

function renderRows() {
  const events = [
    { name: "GDP (Q1)", actual: 27000, previous: 26500, yoy_change: 4.25, unit: "USD billions" },
    { name: "GDP (Q2)", actual: null, previous: 0, yoy_change: null, unit: "USD billions" },
    { name: "FOMC", actual: 3.63, previous: 4, yoy_change: null, unit: "%" },
    { name: "CPI", actual: 333, previous: 332, yoy_change: -2, unit: "% YoY" },
  ];
  const jsx = (type, props) => ({ type, props });
  const passthrough = ({ children }) => children;
  const modules = {
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "@/lib/api": {
      getEconomicCalendar: async () => ({ events, warnings: [] }),
      getFreshness: async () => ({}),
    },
    "@/components/Topbar": { default: () => null },
    "@/components/Explainer": { default: () => null },
    "@/components/ui": { Panel: passthrough, Section: passthrough, fmtAsOf: () => "" },
  };
  const source = ts.transpileModule(page, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const exportsObject = {};
  new Function("require", "exports", source)((name) => {
    if (!Object.hasOwn(modules, name)) throw new Error("Unmapped module " + name);
    return modules[name];
  }, exportsObject);
  function children(node) {
    return Array.isArray(node) ? node : [node];
  }
  function find(node, tag) {
    if (Array.isArray(node)) return node.flatMap((x) => find(x, tag));
    if (!node || typeof node !== "object") return [];
    if (typeof node.type === "function") return find(node.type(node.props), tag);
    return (node.type === tag ? [node] : []).concat(find(node.props.children, tag));
  }
  return exportsObject.default().then((root) => {
    const rows = find(root, "tbody").flatMap((body) => find(body, "tr"));
    return rows.map((row) =>
      find(row, "td")
        .slice(3)
        .map((cell) => children(cell.props.children).flat(Infinity).join("")),
    );
  });
}

test("GDP level cells and controls", async () => {
  const rows = await renderRows();
  const expected = [
    ["27,000 USD billions", "26,500 USD billions", "+4.25%"],
    ["—", "0 USD billions", "—"],
    ["3.63%", "4.00%", "—"],
    ["333", "332", "-2.00%"],
  ];
  assert.deepEqual(rows, expected);
});
