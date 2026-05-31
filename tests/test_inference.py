"""Unit tests for inference module scaffolding and schemas."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.inference.benchmark import BenchmarkConfig, InferenceBenchmark
from src.inference.generate import GenerationConfig, GenerationEngine
from src.serving.api import create_app
from src.serving.schemas import GenerateRequest, HealthResponse


# ── FastAPI tests ─────────────────────────────────────────────────────────────


@pytest.fixture()
def api_client() -> TestClient:
    """Create a ``TestClient`` backed by the FastAPI application."""
    return TestClient(create_app())


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, api_client: TestClient) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, api_client: TestClient) -> None:
        data = api_client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, api_client: TestClient) -> None:
        data = api_client.get("/health").json()
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestGenerateEndpoint:
    """Tests for POST /generate."""

    def test_generate_returns_200(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/generate",
            json={"prompt": "Hello, world!", "max_new_tokens": 64},
        )
        assert response.status_code == 200

    def test_generate_returns_text_field(self, api_client: TestClient) -> None:
        data = api_client.post(
            "/generate",
            json={"prompt": "Tell me a joke."},
        ).json()
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0

    def test_generate_returns_token_counts(self, api_client: TestClient) -> None:
        data = api_client.post(
            "/generate",
            json={"prompt": "Short prompt"},
        ).json()
        assert data["prompt_tokens"] >= 0
        assert data["completion_tokens"] >= 0
        assert data["total_tokens"] == data["prompt_tokens"] + data["completion_tokens"]

    def test_generate_no_input_returns_400(self, api_client: TestClient) -> None:
        response = api_client.post("/generate", json={})
        assert response.status_code == 422  # Pydantic validation error.

    def test_generate_finish_reason_present(self, api_client: TestClient) -> None:
        data = api_client.post(
            "/generate",
            json={"prompt": "Explain AI."},
        ).json()
        assert data["finish_reason"] in ("stop", "length", "error")


# ── GenerationConfig tests ────────────────────────────────────────────────────


class TestGenerationConfig:
    """Tests for GenerationConfig defaults and constraints."""

    def test_default_max_new_tokens(self) -> None:
        config = GenerationConfig()
        assert config.max_new_tokens == 512

    def test_default_temperature(self) -> None:
        config = GenerationConfig()
        assert config.temperature == pytest.approx(0.7)

    def test_default_stop_sequences_is_empty(self) -> None:
        config = GenerationConfig()
        assert config.stop_sequences == []


# ── BenchmarkConfig tests ─────────────────────────────────────────────────────


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig defaults."""

    def test_default_num_requests(self) -> None:
        config = BenchmarkConfig()
        assert config.num_requests == 100

    def test_default_concurrency(self) -> None:
        config = BenchmarkConfig()
        assert config.concurrency == 1


class TestInferenceBenchmarkInit:
    """Tests for InferenceBenchmark construction."""

    def test_init_with_callable(self) -> None:
        bench = InferenceBenchmark(generate_fn=lambda p: "response")
        assert bench._config.num_requests == 100

    def test_custom_config(self) -> None:
        config = BenchmarkConfig(num_requests=10, concurrency=4)
        bench  = InferenceBenchmark(generate_fn=lambda p: "", config=config)
        assert bench._config.num_requests == 10
        assert bench._config.concurrency == 4
