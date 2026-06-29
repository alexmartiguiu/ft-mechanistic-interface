"""Domain exceptions — translated to HTTP status by the API layer."""
from __future__ import annotations


class NotFoundError(Exception):
    """A requested entity does not exist."""

    def __init__(self, entity: str, key: object) -> None:
        super().__init__(f"{entity} {key!r} not found")
        self.entity = entity
        self.key = key
