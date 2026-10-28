export type VixTermPoint = {
  maturity: string;
  days: number;
  value: number;
  symbol: string;
  change: number | null;
  change_pct: number | null;
  estimated: boolean;
};

type Props = {
  term: VixTermPoint[];
  structure: string | null;
  source: string;
  asOf: string;
};

function signed(value: number | null, suffix = "") {
  if (value == null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}${suffix}`;
}

function structureLabel(structure: string | null) {
  if (structure === "Contango") return "Upward sloping";
  if (structure === "Backwardation") return "Inverted";
  return structure ?? "Undetermined";
}

export default function VixTermChart({ term, structure, source, asOf }: Props) {
  if (term.length === 0) {
    return <div className="empty-state">VIX index curve unavailable.</div>;
  }

  const vix = term.find((point) => point.symbol === "VIX");
  const front = term.find((point) => point.symbol === "VIX9D");
  const back = term.find((point) => point.symbol === "VIX1Y");
  const fullSlope = front && back ? back.value - front.value : null;
  const values = term.map((point) => point.value);
  const yMin = Math.floor(Math.min(...values) - 1);
  const yMax = Math.ceil(Math.max(...values) + 1);
  const tone = structure === "Backwardation" ? "bad" : structure === "Contango" ? "good" : "neutral";

  const width = 760;
  const height = 300;
  const left = 48;
  const right = 24;
  const top = 24;
  const bottom = 42;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const yRange = Math.max(yMax - yMin, 1);
  const canonicalDays = [9, 30, 90, 180, 365];
  const xFor = (days: number) => left + ((days - canonicalDays[0]) / (canonicalDays[canonicalDays.length - 1] - canonicalDays[0])) * chartWidth;
  const yFor = (value: number) => top + ((yMax - value) / yRange) * chartHeight;
  const coordinates = term.map((point) => ({ point, x: xFor(point.days), y: yFor(point.value) }));
  const segments = coordinates.reduce<Array<typeof coordinates>>((groups, coordinate) => {
    const previousGroup = groups[groups.length - 1];
    const previousPoint = previousGroup?.[previousGroup.length - 1]?.point;
    const previousIndex = previousPoint ? canonicalDays.indexOf(previousPoint.days) : -1;
    const currentIndex = canonicalDays.indexOf(coordinate.point.days);
    if (!previousGroup || previousIndex < 0 || currentIndex !== previousIndex + 1) {
      groups.push([coordinate]);
    } else {
      previousGroup.push(coordinate);
    }
    return groups;
  }, []);
  const ticks = Array.from({ length: 5 }, (_, index) => yMin + (index * yRange) / 4).reverse();

  return (
    <section className="panel vix-panel" aria-labelledby="vix-term-heading">
      <div className="vix-heading">
        <div>
          <div className="eyebrow">VOLATILITY CURVE</div>
          <h2 id="vix-term-heading">VIX Index Term Structure</h2>
          <p>Index closes — not VIX futures</p>
        </div>
        <div className="vix-meta">
          <span className={`pill ${tone}`}>{structureLabel(structure)}</span>
          <span>{asOf}</span>
        </div>
      </div>

      <div className="vix-summary">
        <div className="vix-stat">
          <span>VIX (30D)</span>
          <strong>{vix ? vix.value.toFixed(2) : "—"}</strong>
          <small>Latest available 30-day index close</small>
        </div>
        <div className="vix-stat">
          <span>9D → 1Y slope</span>
          <strong className={fullSlope != null && fullSlope < 0 ? "negative" : "positive"}>
            {signed(fullSlope)}
          </strong>
          <small>{front && back ? `${front.symbol} to ${back.symbol}` : "Requires both endpoint closes"}</small>
        </div>
        <div className="vix-stat vix-source-stat">
          <span>Source</span>
          <strong>{source}</strong>
          <small>Daily closes; observation dates may differ · response {asOf}</small>
        </div>
      </div>

      <div className="vix-chart-frame">
        <svg className="vix-curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`VIX index curve is ${structureLabel(structure).toLowerCase()}`}>
          <defs>
            <linearGradient id="vixLine" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#d97706" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
          </defs>
          {ticks.map((tick) => {
            const y = yFor(tick);
            return (
              <g key={tick}>
                <line x1={left} x2={width - right} y1={y} y2={y} className="vix-grid-line" />
                <text x={left - 9} y={y + 4} textAnchor="end" className="vix-axis-label">{tick.toFixed(0)}</text>
              </g>
            );
          })}
          {vix && (
            <g>
              <line x1={left} x2={width - right} y1={yFor(vix.value)} y2={yFor(vix.value)} className="vix-reference-line" />
              <text x={width - right} y={yFor(vix.value) - 7} textAnchor="end" className="vix-reference-label">VIX {vix.value.toFixed(2)}</text>
            </g>
          )}
          {segments.filter((segment) => segment.length > 1).map((segment) => (
            <polyline
              key={`${segment[0].point.symbol}-${segment[segment.length - 1].point.symbol}`}
              points={segment.map(({ x, y }) => `${x},${y}`).join(" ")}
              className="vix-curve-line"
            />
          ))}
          {coordinates.map(({ point, x, y }) => (
            <g key={point.symbol}>
              <circle cx={x} cy={y} r="6" className="vix-point">
                <title>{`${point.symbol}: ${point.value.toFixed(2)}`}</title>
              </circle>
              <text x={x} y={height - 17} textAnchor="middle" className="vix-x-label">{point.maturity}</text>
              <text x={x} y={y - 12} textAnchor="middle" className="vix-value-label">{point.value.toFixed(2)}</text>
            </g>
          ))}
        </svg>
      </div>

      <div className="vix-table-wrap">
        <table className="vix-table">
          <thead>
            <tr>
              <th>Tenor</th>
              <th>Index</th>
              <th className="num">Close</th>
              <th className="num">Difference</th>
              <th className="num">Vs prior</th>
            </tr>
          </thead>
          <tbody>
            {term.map((point) => (
              <tr key={point.symbol}>
                <td><b>{point.maturity}</b><span>{point.days}d</span></td>
                <td className="mono">{point.symbol}{point.estimated && <em className="estimated-tag">Est.</em>}</td>
                <td className="num mono">{point.value.toFixed(2)}</td>
                <td className={`num mono ${point.change != null && point.change < 0 ? "negative" : "positive"}`}>{signed(point.change)}</td>
                <td className={`num mono ${point.change_pct != null && point.change_pct < 0 ? "negative" : "positive"}`}>{signed(point.change_pct, "%")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
