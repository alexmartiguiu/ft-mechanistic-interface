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
