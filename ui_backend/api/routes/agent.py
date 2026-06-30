"""Agent session routes — SSE down, POST up.

The agent runs server-side; the browser opens an EventSource on `/stream` and
POSTs answers / action-clicks / messages back. Events are the M1.1 typed contract,
serialized one per SSE frame with `event: <kind>`.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ui_backend.agent import manager
from ui_backend.core.database import SessionLocal
from ui_backend.models.run import Run
from ui_backend.services.live_run import LiveRunService

router = APIRouter(prefix="/agent", tags=["agent"])


class SessionCreate(BaseModel):
    run_id: int
    mode: str | None = None  # advisory only; the server derives mode from the run's project
    model_use: str | None = None  # "what does this model do in the world" → injected into the prompt


class CreateRunIn(BaseModel):
    domain: str
    model_id: str
    lora_preset: str | None = None
    concepts: list[str] | None = None


def _run_mode(run_id: int) -> str:
    """Authoritative mode for a session: the run's project decides replay vs live."""
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run.project.mode or "replay"


@router.post("/create-run", status_code=201)
def create_run(payload: CreateRunIn):
    """Create a live run (rows + launch command) and return its id; the front-end then
    opens a session on it. Reuses curated domain configs + pre-minted vectors (v1)."""
    with SessionLocal() as db:
        return LiveRunService(db).create_run(
            domain=payload.domain, model_id=payload.model_id,
            lora_preset=payload.lora_preset, concepts=payload.concepts,
        )


class MessageIn(BaseModel):
    text: str


class AnswerIn(BaseModel):
    ref: str | None = None
    value: object = None  # str (single) or list[str] (multi)


class ActionIn(BaseModel):
    ref: str | None = None


@router.post("/sessions", status_code=201)
async def create_session(payload: SessionCreate):
    mode = _run_mode(payload.run_id)  # per-project truth, not the client's hint
    try:
        sess = await manager.create(payload.run_id, mode=mode, model_use=payload.model_use)
    except Exception as e:  # noqa: BLE001 — surface SDK/Bedrock/DB startup failures cleanly
        raise HTTPException(500, f"agent failed to start: {e}") from e
    return {"sid": sess.sid, "run_id": sess.run_id, "mode": sess.mode}


@router.get("/sessions/{sid}/stream")
async def stream(sid: str):
    sess = manager.get(sid)
    if sess is None:
        raise HTTPException(404, "no such session")

    async def gen():
        # default "message" frames (kind is in the JSON) → one EventSource.onmessage handler
        while not sess.closed:
            ev = await sess.events.get()
            yield f"data: {ev.model_dump_json()}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/sessions/{sid}/message")
async def message(sid: str, payload: MessageIn):
    sess = manager.get(sid)
    if sess is None:
        raise HTTPException(404, "no such session")
    await sess.send(payload.text)
    return {"ok": True}


@router.post("/sessions/{sid}/intent")
async def intent(sid: str, payload: MessageIn):
    """Add 'what this model does in the world' after the session started (replay): it
    rides the agent's next turn rather than the (already-frozen) system prompt."""
    sess = manager.get(sid)
    if sess is None:
        raise HTTPException(404, "no such session")
    sess.set_model_use(payload.text)
    return {"ok": True}


@router.post("/sessions/{sid}/answer")
async def answer(sid: str, payload: AnswerIn):
    sess = manager.get(sid)
    if sess is None:
        raise HTTPException(404, "no such session")
    return {"ok": sess.resolve(payload.ref or "", payload.value)}


@router.post("/sessions/{sid}/action")
async def action(sid: str, payload: ActionIn):
    sess = manager.get(sid)
    if sess is None:
        raise HTTPException(404, "no such session")
    return {"ok": sess.resolve(payload.ref or "", "clicked")}


@router.delete("/sessions/{sid}", status_code=204)
async def close_session(sid: str):
    await manager.close(sid)
