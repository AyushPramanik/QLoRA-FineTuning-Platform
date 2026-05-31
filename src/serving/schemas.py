"""Pydantic request and response schemas for the inference API.

All API contracts are defined here so that both the FastAPI router and
client code can import from a single source of truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Chat message role."""

    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role:    Role   = Field(..., description="The role of the message author.")
    content: str    = Field(..., min_length=1, description="Message content.")


# ── Request schemas ────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for the ``POST /generate`` endpoint.

    Attributes:
        prompt:             Raw text prompt (mutually exclusive with ``messages``).
        messages:           Chat messages (mutually exclusive with ``prompt``).
        max_new_tokens:     Maximum tokens to generate.
        temperature:        Sampling temperature. Set to 0 for greedy decoding.
        top_p:              Nucleus sampling probability mass.
        top_k:              Top-K sampling cutoff.
        repetition_penalty: Penalty for repeated n-grams.
        stream:             Whether to stream the response token-by-token.
        stop:               List of stop strings.
    """

    prompt:             str | None          = Field(None, description="Raw text prompt.")
    messages:           list[ChatMessage] | None = Field(None, description="Chat history.")
    max_new_tokens:     int                 = Field(256, ge=1, le=8192)
    temperature:        float               = Field(0.7, ge=0.0, le=2.0)
    top_p:              float               = Field(0.9, ge=0.0, le=1.0)
    top_k:              int                 = Field(50, ge=1, le=1000)
    repetition_penalty: float               = Field(1.1, ge=1.0, le=2.0)
    stream:             bool                = Field(False)
    stop:               list[str]           = Field(default_factory=list)

    @field_validator("prompt", "messages", mode="before")
    @classmethod
    def exactly_one_input(cls, v: Any, info: Any) -> Any:
        """Ensure exactly one of prompt or messages is provided."""
        # TODO: Implement cross-field validation with model_validator.
        return v

    model_config = {"json_schema_extra": {
        "examples": [{
            "prompt": "Explain quantum entanglement in simple terms.",
            "max_new_tokens": 256,
            "temperature": 0.7,
        }]
    }}


# ── Response schemas ───────────────────────────────────────────────────────────


class GenerateResponse(BaseModel):
    """Response body for the ``POST /generate`` endpoint.

    Attributes:
        text:              The generated completion text.
        prompt_tokens:     Number of tokens in the input prompt.
        completion_tokens: Number of generated tokens.
        total_tokens:      Sum of prompt and completion tokens.
        finish_reason:     Why generation stopped (``"stop"``, ``"length"``, ``"error"``).
        model:             Identifier of the model that produced the response.
        latency_ms:        End-to-end request latency in milliseconds.
    """

    text:              str   = Field(..., description="Generated completion.")
    prompt_tokens:     int   = Field(..., ge=0)
    completion_tokens: int   = Field(..., ge=0)
    total_tokens:      int   = Field(..., ge=0)
    finish_reason:     str   = Field(..., description="Reason generation stopped.")
    model:             str   = Field(..., description="Serving model identifier.")
    latency_ms:        float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status:  str = "ok"
    version: str = "0.1.0"
    model:   str | None = None


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error:   str = Field(..., description="Error type or code.")
    message: str = Field(..., description="Human-readable error description.")
    detail:  Any | None = None
