"""FastAPI application factory.

Run locally:

    uvicorn app.main:app --reload --port 8000

Interactive OpenAPI docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.v1.router import router as v1_router
from app.config import get_settings
from app.db.base import init_db
from app.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Agentic customer-support assistant for an e-commerce retailer: "
            "LangGraph routing (direct / RAG / tools), Pinecone-backed retrieval, "
            "customer-scoped order tools, persistent chat sessions."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(v1_router)

    # Back-compat alias so /api/health works as well as /api/v1/health.
    @app.get("/api/health", include_in_schema=False)
    def health_alias():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/api/v1/health")

    return app


app = create_app()
