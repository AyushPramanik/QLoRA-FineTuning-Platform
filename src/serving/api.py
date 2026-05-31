"""FastAPI application — inference REST API.

Exposes:
  GET  /health    — liveness probe
  POST /generate  — text completion (mock response for scaffolding)

This module is intentionally free of actual model loading; the real
inference engine will be injected via FastAPI dependency injection.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.serving.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# ── Application state ─────────────────────────────────────────────────────────

_MODEL_NAME: str = "placeholder-model-v0.1"
_SERVING_MODEL: Any = None  # Will hold the inference engine in production.


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events.

    On startup: configure logging and load the model.
    On shutdown: release GPU memory.
    """
    # TODO: Load the inference engine from environment config.
    # TODO: Register the batching engine background task.
    configure_logging()
    logger.info("api_startup", model=_MODEL_NAME)
    yield
    logger.info("api_shutdown")
    # TODO: Unload model and free CUDA memory.


# ── Application factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured ``FastAPI`` instance.
    """
    app = FastAPI(
        title="QLoRA Fine-Tuning Platform — Inference API",
        description="REST API for text generation with fine-tuned language models.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # TODO: Restrict to known origins in production.
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Register exception handlers.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred.",
            ).model_dump(),
        )

    # Register routers.
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Attach route handlers to the application."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness probe",
        tags=["Operations"],
    )
    async def health() -> HealthResponse:
        """Return service health status.

        Returns:
            ``{"status": "ok", "version": "0.1.0", "model": "..."}``
        """
        return HealthResponse(status="ok", version="0.1.0", model=_MODEL_NAME)

    @app.post(
        "/generate",
        response_model=GenerateResponse,
        summary="Generate text completion",
        tags=["Inference"],
        status_code=status.HTTP_200_OK,
    )
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """Generate a text completion for the given prompt.

        This endpoint returns a **mock response** until the real inference
        engine is implemented.

        Args:
            request: ``GenerateRequest`` body with prompt and generation params.

        Returns:
            ``GenerateResponse`` with the generated text and token counts.

        Raises:
            HTTPException 400: If neither prompt nor messages is provided.
            HTTPException 503: If the model is not loaded.
        """
        if request.prompt is None and request.messages is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'prompt' or 'messages' must be provided.",
            )

        # TODO: Replace mock response with real inference engine call.
        # TODO: Implement streaming via StreamingResponse when request.stream=True.
        # TODO: Route to DynamicBatchingEngine for batched GPU execution.

        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        logger.info(
            "generate_request",
            request_id=request_id,
            max_new_tokens=request.max_new_tokens,
            stream=request.stream,
        )

        # ── Mock response (placeholder) ───────────────────────────────────────
        mock_text = (
            "This is a placeholder response. "
            "The actual inference engine is not yet implemented. "
            "Configure a model and replace this stub with a real generation call."
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return GenerateResponse(
            text=mock_text,
            prompt_tokens=len((request.prompt or "").split()),
            completion_tokens=len(mock_text.split()),
            total_tokens=len((request.prompt or "").split()) + len(mock_text.split()),
            finish_reason="stop",
            model=_MODEL_NAME,
            latency_ms=round(latency_ms, 2),
        )


# ── Module-level app instance (for uvicorn) ───────────────────────────────────

app = create_app()


def main() -> None:
    """CLI entrypoint — launch the server with uvicorn."""
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
