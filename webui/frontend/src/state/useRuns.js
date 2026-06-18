import { useCallback, useEffect, useRef, useState } from "react";
import { runScript } from "../data/demo.js";

// Owns the list of runs and drives their (replayed) log streams on a timer.
// Ported from the prototype's startRun/stopRun. This is the demo/replay path
// (PLAN.md 2.6); the live path will swap the timer for api.streamRun(). The run
// list is the first-class object both the Runs screen and New Experiment share.
export function useRuns({ stepMs = 360 } = {}) {
  const [runs, setRuns] = useState([]);
  const seq = useRef(0);
  const timers = useRef({});

  useEffect(() => () => Object.values(timers.current).forEach(clearInterval), []);

  const startRun = useCallback((cmd, app, gpu, extra) => {
    seq.current += 1;
    const id = "run" + seq.current;
    const argv = `python -m hedda ${cmd} --app ${app} --gpu ${gpu}${extra ? " " + extra : ""}`;
    const script = runScript(cmd, app, gpu, extra);
    const run = {
      id, status: "running", argv, started: Date.now() / 1000, returncode: null,
      cmd, app, _script: script, _idx: 0, lines: [],
    };
    setRuns((rs) => [run, ...rs]);
    timers.current[id] = setInterval(() => {
      setRuns((rs) =>
        rs.map((r) => {
          if (r.id !== id || r.status !== "running") return r;
          if (r._idx >= r._script.length) {
            clearInterval(timers.current[id]);
            return { ...r, status: "done", returncode: 0 };
          }
          return { ...r, lines: r.lines.concat([r._script[r._idx]]), _idx: r._idx + 1 };
        })
      );
    }, stepMs);
    return id;
  }, [stepMs]);

  const stopRun = useCallback((id) => {
    clearInterval(timers.current[id]);
    setRuns((rs) =>
      rs.map((r) =>
        r.id === id
          ? { ...r, status: "failed", returncode: 130, lines: r.lines.concat(["[stopped] cancelled by user"]) }
          : r
      )
    );
  }, []);

  return { runs, startRun, stopRun };
}
