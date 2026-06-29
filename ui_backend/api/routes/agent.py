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

router = APIRouter(prefix="/agent", tags=["agent"])


class SessionCreate(BaseModel):
    run_id: int
    mode: str = "replay"


class MessageIn(BaseModel):
    text: str


class AnswerIn(BaseModel):
    ref: str | None = None
    value: object = None  # str (single) or list[str] (multi)


class ActionIn(BaseModel):
    ref: str | None = None


@router.post("/sessions", status_code=201)
async def create_session(payload: SessionCreate):
    try:
        sess = await manager.create(payload.run_id, mode=payload.mode)
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
