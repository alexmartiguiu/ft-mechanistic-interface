import { useEffect, useRef, useState } from "react";

// Measure an element's width + height (responsive SVG charts read these).
export function useSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setSize({ w: cr.width, h: cr.height });
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

// Animate elapsed milliseconds 0→`total` (via rAF) once `active` turns true, then
// hold at `total`. Returns 0 while inactive. Unlike useReveal this exposes the raw
// clock so a caller can drive SEVERAL staged reveals off one timeline (e.g. fill the
// loss lines, then the eval lines, then the projections, each in its own [a,b] window).
export function useTimeline(active, total = 5000) {
  const [t, setT] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    if (!active) { setT(0); return; }
    setT(0);
    const t0 = performance.now();
    const tick = (now) => {
      const e = Math.min(total, now - t0);
      setT(e);
      if (e < total) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [active, total]);
  return t;
}

// clamp((t-a)/(b-a), 0, 1): the 0→1 fill fraction for a stage occupying [a,b] ms.
export function windowFrac(t, a, b) {
  return Math.max(0, Math.min(1, (t - a) / (b - a || 1)));
}

// Animate a 0→1 "reveal" fraction over `duration` ms when `active` turns true.
// Used to fill the insight plots left-to-right as if running live. `idle` is the
// value held while inactive: 1 for a recorded plot that should read as complete,
// but 0 for a plot that mounts BEFORE its live-fill starts (e.g. the steered plot,
// which appears during the morph and must show empty axes until streaming begins).
export function useReveal(active, duration = 5000, idle = 1) {
  const [r, setR] = useState(active ? 0 : idle);
  const raf = useRef(0);
  useEffect(() => {
    if (!active) { setR(idle); return; }
    setR(0);
    const t0 = performance.now();
    const tick = (t) => {
      const f = Math.min(1, (t - t0) / duration);
      setR(f);
      if (f < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [active, duration, idle]);
  return r;
}
