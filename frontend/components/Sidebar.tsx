"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

type Item = { label: string; icon: ReactNode; href?: string };
type Group = { label: string; items: Item[] };

const I = {
  grid: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></svg>
  ),
  wave: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 15l4-6 4 4 5-9 5 12" /></svg>,
  bars: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 20V10M10 20V4M16 20v-8M22 20H2" /></svg>,
  flow: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 17l6-6 4 4 8-8" /><path d="M21 3v6h-6" /></svg>,
  globe: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 010 18M12 3a14 14 0 000 18" /></svg>,
  gear: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 13a7.9 7.9 0 000-2l2-1.5-2-3.5-2.4 1a7.9 7.9 0 00-1.7-1L14.9 3H10l-.4 2.5a7.9 7.9 0 00-1.7 1l-2.4-1-2 3.5 2 1.5a7.9 7.9 0 000 2l-2 1.5 2 3.5 2.4-1a7.9 7.9 0 001.7 1L10 21h4.9l.4-2.5a7.9 7.9 0 001.7-1l2.4 1 2-3.5z" /></svg>,
};

const GROUPS: Group[] = [
  { label: "Volatility", items: [
    { label: "Volatility & VRP", icon: I.wave, href: "/volatility" },
    { label: "Sectors & VIX", icon: I.wave },
    { label: "Treasury Stress", icon: I.bars },
  ]},
  { label: "Credit & Rates", items: [
    { label: "Credit & Liquidity", icon: I.flow },
    { label: "Repo Market", icon: I.bars },
    { label: "Fed Watch", icon: I.bars },
  ]},
  { label: "Positioning", items: [
    { label: "COT Positioning", icon: I.flow },
    { label: "CTA Flow", icon: I.flow },
    { label: "Options Flow", icon: I.wave },
    { label: "Institutional Flow", icon: I.bars },
  ]},
  { label: "Macro & Breadth", items: [
    { label: "Cross-Asset", icon: I.globe },
    { label: "Economic Calendar", icon: I.bars },
    { label: "Market Breadth", icon: I.wave, href: "/breadth" },
    { label: "LEFT Strategy", icon: I.flow },
    { label: "Sentiment", icon: I.globe },
  ]},
  { label: "System", items: [
    { label: "Settings", icon: I.gear },
    { label: "System Health", icon: I.wave },
  ]},
];

function NavLink({ item, active }: { item: Item; active: boolean }) {
  if (!item.href) {
    return (
      <a className="nav-item soon" aria-disabled="true" title="Available in a later phase">
        {item.icon}{item.label}
      </a>
    );
  }
  return (
    <Link className={`nav-item${active ? " active" : ""}`} href={item.href} aria-current={active ? "page" : undefined}>
      {item.icon}{item.label}
    </Link>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-hidden />
        <div>
          <span className="brand-name">Meridian</span>
          <span className="brand-sub">Market Risk</span>
        </div>
      </div>
      <nav className="nav" aria-label="Primary">
        <NavLink item={{ label: "Overview", icon: I.grid, href: "/" }} active={pathname === "/"} />
        {GROUPS.map((g) => (
          <div className="nav-group" key={g.label}>
            <div className="nav-group-label">{g.label}</div>
            {g.items.map((it) => (
              <NavLink key={it.label} item={it} active={!!it.href && pathname === it.href} />
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-foot">
        Not financial advice. Data: FRED, CBOE, Yahoo Finance, Fed, Treasury.
      </div>
    </aside>
  );
}
