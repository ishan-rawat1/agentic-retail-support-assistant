"""Centralized exception handling: domain errors -> structured HTTP responses."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.exceptions import AuthenticationError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthenticationError)
    async def auth_error(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": str(exc), "code": "unauthorized"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc), "code": "not_found"})

    @app.exception_handler(ValidationError)
    async def validation(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc), "code": "validation_error"})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "code": "internal_error"},
        )
