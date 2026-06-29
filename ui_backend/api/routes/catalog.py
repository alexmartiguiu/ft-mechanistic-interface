"""Catalog endpoints (Setup screen)."""
from __future__ import annotations

from fastapi import APIRouter

from ui_backend import schemas
from ui_backend.api.deps import CatalogServiceDep

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("", response_model=schemas.CatalogRead)
def get_catalog(service: CatalogServiceDep) -> schemas.CatalogRead:
    return service.get_catalog()
