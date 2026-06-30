"""JobManager — own the live `ftmi` GPU subprocesses, decoupled from any session.

A live run is a long (≈hours) `ftmi run` subprocess. It must outlive the browser/SSE
connection and the agent session that kicked it off, so it is owned here at process
scope (mirroring `agent.AgentSessionManager`), keyed by `run_id`:

  • `ensure(run_id)`     — idempotent: start the job, or attach to a running one.
  • `subscribe(run_id)`  — a fresh event queue; replays cached audit/curves so a late
                            joiner's plots catch up, then streams live frames.
  • `wait_audit(run_id)` — resolves the moment the early audit.json snapshot lands.
  • `cancel(run_id)`     — kill the whole process group (the orchestrator forks children).

The per-job `_watch` coroutine polls `data/<name>/` as checkpoints/eval land, syncs the
DB via `persist_run`, and broadcasts the SAME typed events the agent tools emit
(StageEvent / MetricEvent) — so the front-end can't tell a live fill from a replay one.

In-memory only (v1): a server restart loses the handles and orphans a running `ftmi`;
reconcile-on-startup is a follow-up.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from ui_backend.core.config import REPO_ROOT, Settings, get_settings
from ui_backend.core.database import SessionLocal
from ui_backend.db.ingest import persist_run
from ui_backend.models.enums import RunStatus
from ui_backend.models.project import Project
from ui_backend.models.run import Run
from ui_backend.schemas.events import MetricEvent, StageEvent
from ui_backend.services.providers.replay import ReplayProvider

log = logging.getLogger("ui_backend.jobs")

# Physical CUDA devices the backend may schedule on; override with FTMI_UI_GPUS="0,2".
# With ≥2 free, per-checkpoint eval pipelines concurrently with training.
_GPUS: list[str] = [g for g in os.environ.get("FTMI_UI_GPUS", "0,2").split(",") if g.strip()]

POLL_SECONDS = 12          # how often the watcher syncs disk → DB → events
DONE = object()            # sentinel pushed onto a subscriber queue when the job ends


@dataclass
class JobSpec:
    """Everything needed to launch + persist a live run, recovered from Run.argv."""
    run_id: int
    name: str                       # dir_slug → data/<name>/
    project_id: int
    domain: str
    base_model_id: str
    app_config_base: str            # curated config the run was launched from (medical, …)
    cmd: list[str]                  # `python -m ftmi.cli run --app … --name … --skip-vectors …`
                                    # (WITHOUT gpu flags — JobManager appends them at spawn)

    @classmethod
    def from_run(cls, run: Run) -> "JobSpec":
        meta = json.loads(run.argv) if run.argv else {}
        return cls(
            run_id=run.id,
            name=run.dir_slug or meta.get("name"),
            project_id=run.project_id,
            domain=run.project.domain,
            base_model_id=run.base_model_id,
            app_config_base=meta.get("app_config_base") or run.project.domain,
            cmd=list(meta.get("cmd") or []),
        )


@dataclass
class JobHandle:
    spec: JobSpec
    status: RunStatus = RunStatus.queued
    proc: asyncio.subprocess.Process | None = None
    train_gpu: str | None = None
    eval_gpu: str | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    last_curves: dict | None = None
    last_audit: dict | None = None
    audit_ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None
    task: asyncio.Task | None = None
    logfile: object = None
    _audit_emitted: bool = False
    _training_started: bool = False


class JobManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._jobs: dict[int, JobHandle] = {}
        self._free_gpus: list[str] = list(_GPUS)
        self._gpu_lock = asyncio.Lock()

    # ── public surface ───────────────────────────────────────────────────────
    def get(self, run_id: int) -> JobHandle | None:
        return self._jobs.get(run_id)

    async def ensure(self, run_id: int) -> JobHandle:
        """Start the job for `run_id`, or return the already-running one (idempotent)."""
        job = self._jobs.get(run_id)
        if job is not None:
            return job
        with SessionLocal() as db:
            run = db.get(Run, run_id)
            if run is None:
                raise ValueError(f"run {run_id} not found")
            spec = JobSpec.from_run(run)
        if not spec.cmd:
            raise ValueError(f"run {run_id} has no launch command (argv); was it created live?")
        job = JobHandle(spec=spec)
        self._jobs[run_id] = job
        job.task = asyncio.create_task(self._run(job))
        return job

    def subscribe(self, run_id: int) -> asyncio.Queue:
        """A fresh queue, pre-loaded with the latest cached frames so a late joiner catches up."""
        job = self._jobs[run_id]
        q: asyncio.Queue = asyncio.Queue()
        if job.last_audit is not None:
            q.put_nowait(StageEvent(kind="audit_flagged", view="audit", payload=job.last_audit))
        if job.last_curves is not None:
            q.put_nowait(StageEvent(kind="training_started", view="insights"))
            q.put_nowait(StageEvent(kind="training_fill", view="insights", payload=job.last_curves))
        if job.done.is_set():
            q.put_nowait(DONE)
        else:
            job.subscribers.add(q)
        return q

    def unsubscribe(self, run_id: int, q: asyncio.Queue) -> None:
        job = self._jobs.get(run_id)
        if job is not None:
            job.subscribers.discard(q)

    async def wait_audit(self, run_id: int) -> None:
        await self._jobs[run_id].audit_ready.wait()

    async def cancel(self, run_id: int) -> None:
        """Request cancellation; `_run`'s finally does the DB flip + cleanup."""
        job = self._jobs.get(run_id)
        if job is None or job.done.is_set():
            return
        job.status = RunStatus.canceled
        self._kill(job)
        if job.task is not None:
            job.task.cancel()

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def _run(self, job: JobHandle) -> None:
        try:
            await self._acquire_gpus(job)
            await self._spawn(job)
            await self._mark_started(job)
            waiter = asyncio.ensure_future(job.proc.wait())
            while not waiter.done():
                self._tick(job)
                try:
                    # shield so the POLL timeout never cancels the underlying wait()
                    await asyncio.wait_for(asyncio.shield(waiter), timeout=POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
            rc = job.proc.returncode
            if job.status != RunStatus.canceled:
                job.status = RunStatus.done if rc == 0 else RunStatus.failed
                if rc != 0:
                    job.error = job.error or f"ftmi exited rc={rc}"
        except asyncio.CancelledError:
            job.status = RunStatus.canceled
            self._kill(job)
        except Exception as exc:  # noqa: BLE001 — surface launch/watcher failures as a failed run
            job.status = RunStatus.failed
            job.error = str(exc)
            log.exception("live job %s failed", job.spec.name)
        finally:
            self._finish_db(job, job.status)
            self._tick(job, final=True)
            if job.status == RunStatus.failed:
                self._broadcast(job, StageEvent(kind="status", payload={"error": job.error}))
            self._release_gpus(job)
            self._close(job)

    async def _acquire_gpus(self, job: JobHandle) -> None:
        """Block (emitting a queued status) until at least one device is free; take up to two."""
        announced = False
        while True:
            async with self._gpu_lock:
                if self._free_gpus:
                    job.train_gpu = self._free_gpus.pop(0)
                    job.eval_gpu = self._free_gpus.pop(0) if self._free_gpus else None
                    return
            if not announced:
                self._broadcast(job, StageEvent(kind="status", payload={"queued": True}))
                announced = True
            await asyncio.sleep(POLL_SECONDS)

    def _release_gpus(self, job: JobHandle) -> None:
        for g in (job.train_gpu, job.eval_gpu):
            if g is not None and g not in self._free_gpus:
                self._free_gpus.append(g)
        self._free_gpus.sort()
        job.train_gpu = job.eval_gpu = None

    async def _spawn(self, job: JobHandle) -> None:
        cmd = list(job.spec.cmd)
        if job.train_gpu is not None:
            cmd += ["--train-gpu", job.train_gpu]
        if job.eval_gpu is not None and job.eval_gpu != job.train_gpu:
            cmd += ["--eval-gpu", job.eval_gpu]
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "src")
        log_dir = self.settings.data_root / job.spec.name
        log_dir.mkdir(parents=True, exist_ok=True)
        job.logfile = open(log_dir / "run.log", "a")  # noqa: SIM115 — closed in _close
        log.info("launching live job %s: %s", job.spec.name, " ".join(cmd))
        job.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(REPO_ROOT), env=env,
            stdout=job.logfile, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group → killpg reaps the forked children
        )

    async def _mark_started(self, job: JobHandle) -> None:
        job.status = RunStatus.running
        gpu = ",".join(g for g in (job.train_gpu, job.eval_gpu) if g)
        with SessionLocal() as db:
            run = db.get(Run, job.spec.run_id)
            if run is not None:
                run.status = RunStatus.running
                run.started_at = datetime.now(timezone.utc)
                run.gpu = gpu or None
                db.commit()

    # ── the disk → DB → events tick ──────────────────────────────────────────
    def _tick(self, job: JobHandle, *, final: bool = False) -> None:
        """Sync this run's on-disk state into the DB and broadcast the resulting frames."""
        spec = job.spec
        status = job.status if final else RunStatus.running
        try:
            with SessionLocal() as db:
                project = db.get(Project, spec.project_id)
                if project is None:
                    return
                persist_run(
                    db, spec.name, project=project, domain=spec.domain,
                    base_model_id=spec.base_model_id, status=status,
                    logical_base=spec.name, app_config_base=spec.app_config_base,
                    settings=self.settings,
                )
                rp = ReplayProvider(db, self.settings)
                self._emit_audit(job, rp)
                self._emit_training(job, rp, final=final)
        except Exception:  # noqa: BLE001 — a tick must never kill the watcher loop
            log.exception("tick failed for live job %s", spec.name)

    def _emit_audit(self, job: JobHandle, rp: ReplayProvider) -> None:
        if job._audit_emitted:
            return
        ckpts = self.settings.data_root / job.spec.name / "checkpoints"
        if not (ckpts / "audit.json").exists():
            return
        try:
            au = rp.audit(job.spec.run_id)
        except Exception:  # noqa: BLE001
            return
        if not au.concepts:
            return
        payload = au.model_dump()
        job.last_audit = payload
        job._audit_emitted = True
        self._broadcast(job, StageEvent(kind="audit_flagged", view="audit", payload=payload))
        self._broadcast(job, MetricEvent(
            value=au.total_flagged or 0,
            label=f"samples flagged · p{au.percentile}", tone="bad"))
        job.audit_ready.set()

    def _emit_training(self, job: JobHandle, rp: ReplayProvider, *, final: bool) -> None:
        spec = job.spec
        ckpts = self.settings.data_root / spec.name / "checkpoints"
        has_progress = (ckpts / "progress.jsonl").exists() or any(ckpts.glob("checkpoint-*"))
        if not has_progress and not final:
            return
        try:
            cv = rp.train(spec.run_id)
        except Exception:  # noqa: BLE001
            return
        payload = cv.model_dump()
        job.last_curves = payload
        if not job._training_started:
            job._training_started = True
            self._broadcast(job, StageEvent(kind="training_started", view="insights"))
        self._broadcast(job, StageEvent(kind="training_fill", view="insights", payload=payload))
        worst = self._worst_drift(cv)
        if worst is not None:
            self._broadcast(job, MetricEvent(
                value=round(worst[1], 2), label=f"{worst[0]} probe (worst drift)", tone="bad"))

    @staticmethod
    def _worst_drift(cv) -> tuple[str, float] | None:
        worst = None
        for t in cv.trajectory:
            if t.points:
                p = t.points[-1].probe_prob
                if p is not None and (worst is None or p > worst[1]):
                    worst = (t.concept, p)
        return worst

    # ── helpers ──────────────────────────────────────────────────────────────
    def _broadcast(self, job: JobHandle, ev: BaseModel | object) -> None:
        for q in list(job.subscribers):
            q.put_nowait(ev)

    def _finish_db(self, job: JobHandle, status: RunStatus) -> None:
        if status not in (RunStatus.done, RunStatus.failed, RunStatus.canceled):
            return
        try:
            with SessionLocal() as db:
                run = db.get(Run, job.spec.run_id)
                if run is not None:
                    run.status = status
                    run.finished_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception:  # noqa: BLE001
            log.exception("finish_db failed for %s", job.spec.name)

    def _kill(self, job: JobHandle) -> None:
        if job.proc is not None and job.proc.returncode is None:
            try:
                os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def _close(self, job: JobHandle) -> None:
        job.done.set()
        job.audit_ready.set()  # unblock any waiter even if audit never fired
        self._broadcast(job, DONE)
        job.subscribers.clear()
        if job.logfile is not None:
            try:
                job.logfile.close()
            except Exception:  # noqa: BLE001
                pass
            job.logfile = None


# Process-level singleton (one per backend process), mirroring agent.manager.
jobs = JobManager()
