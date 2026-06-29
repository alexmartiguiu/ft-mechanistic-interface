"""ui_backend — FastAPI data layer for the FTMI front-end.

Layered: models (SQLAlchemy ORM) → repositories (data access) → services
(business logic, returns Pydantic) → api (HTTP). Schemas are the Pydantic
contracts crossing the service/api boundary. See README.md.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
