import { useEffect, useRef, useState } from "react";

// Measure an element's width (responsive SVG charts read this).
export function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setW(cr.width);
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// Animate a 0→1 "reveal" fraction over `duration` ms when `active` turns true.
// Used to fill the insight plots left-to-right as if running live.
export function useReveal(active, duration = 5000) {
  const [r, setR] = useState(active ? 0 : 1);
  const raf = useRef(0);
  useEffect(() => {
    if (!active) { setR(1); return; }
    setR(0);
    const t0 = performance.now();
    const tick = (t) => {
      const f = Math.min(1, (t - t0) / duration);
      setR(f);
      if (f < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [active, duration]);
  return r;
}
