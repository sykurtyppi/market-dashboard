"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { NavigationContent } from "@/components/Sidebar";

const RETURN_FOCUS_KEY = "meridian-mobile-nav-return-focus";

export default function MobileNavigation() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLDivElement>(null);
  const wasOpen = useRef(false);

  const close = () => setOpen(false);

  useEffect(() => {
    if (!open) return;

    const activeLink = drawerRef.current?.querySelector<HTMLElement>('[aria-current="page"]');
    const firstLink = drawerRef.current?.querySelector<HTMLElement>("a[href]");
    const focusTarget = activeLink ?? firstLink ?? drawerRef.current;
    focusTarget?.focus();
    activeLink?.scrollIntoView({ block: "nearest" });

    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(
        drawerRef.current?.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])') ?? [],
      );
      if (focusable.length === 0) {
        event.preventDefault();
        drawerRef.current?.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousBodyOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open && wasOpen.current) triggerRef.current?.focus();
    wasOpen.current = open;
  }, [open]);

  useEffect(() => {
    if (sessionStorage.getItem(RETURN_FOCUS_KEY)) {
      sessionStorage.removeItem(RETURN_FOCUS_KEY);
      requestAnimationFrame(() => triggerRef.current?.focus());
    }
  }, []);

  useEffect(() => {
    const mobileBreakpoint = window.matchMedia("(max-width: 720px)");
    const closeAboveBreakpoint = (event: MediaQueryListEvent) => {
      if (!event.matches) setOpen(false);
    };
    mobileBreakpoint.addEventListener("change", closeAboveBreakpoint);
    return () => mobileBreakpoint.removeEventListener("change", closeAboveBreakpoint);
  }, []);

  return (
    <>
      <button
        ref={triggerRef}
        className="mobile-nav-trigger"
        type="button"
        aria-label="Open navigation"
        aria-expanded={open}
        aria-controls="mobile-navigation"
        onClick={() => setOpen((value) => !value)}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path d="M4 7h16M4 12h16M4 17h16" />
        </svg>
      </button>
      {open && typeof document !== "undefined"
        ? createPortal(
          <div className="mobile-nav-layer">
            <button className="mobile-nav-backdrop" type="button" aria-label="Close navigation" onClick={close} />
            <div
              ref={drawerRef}
              id="mobile-navigation"
              className="mobile-nav-drawer"
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              tabIndex={-1}
            >
              <NavigationContent onNavigate={(href) => {
                if (href !== pathname) sessionStorage.setItem(RETURN_FOCUS_KEY, "true");
                close();
              }} />
            </div>
          </div>,
          document.body,
        )
        : null}
    </>
  );
}