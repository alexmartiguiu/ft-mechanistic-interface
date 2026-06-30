"""Map domain exceptions to HTTP responses (registered in main.py)."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ui_backend.services.exceptions import NotFoundError, PipelineError, ValidationError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(_request: Request, exc: NotFoundError) -> JSONResponse:  # noqa: RUF029
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def _invalid(_request: Request, exc: ValidationError) -> JSONResponse:  # noqa: RUF029
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PipelineError)
    async def _pipeline(_request: Request, exc: PipelineError) -> JSONResponse:  # noqa: RUF029
        return JSONResponse(status_code=500, content={"detail": str(exc)})
