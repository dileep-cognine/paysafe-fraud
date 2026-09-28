"""FastAPI service for the approved fraud-scoring model."""

from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from fraud_scoring.config import AppConfig, load_config
from fraud_scoring.data_validation import ALLOWED_MERCHANT_CATEGORIES
from fraud_scoring.features import FeatureContractError
from fraud_scoring.predict import LoadedChampion, load_champion, score_transaction

logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    """Only authorization-time fields; unknown and training-only fields fail."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: StrictStr = Field(min_length=1)
    amount: Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
    merchant_category: StrictStr
    hour_of_day: Annotated[int, Field(ge=0, le=23, strict=True)]
    device_risk: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]

    @field_validator("transaction_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transaction_id must not be blank")
        return value

    @field_validator("merchant_category")
    @classmethod
    def allowed_merchant(cls, value: str) -> str:
        if value not in ALLOWED_MERCHANT_CATEGORIES:
            raise ValueError("merchant_category is not an allowed category")
        return value


class ScoreResponse(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create an API whose lifespan loads one pinned approved model version."""
    selected = config or load_config()
    logger.setLevel(selected.server.log_level.upper())

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            application.state.champion = load_champion(selected)
        except Exception as exc:
            logger.exception("champion_model_load_failed")
            raise RuntimeError("Champion model unavailable; API startup aborted.") from exc
        logger.info("champion_model_loaded version=%s", application.state.champion.version)
        try:
            yield
        finally:
            application.state.champion = None

    application = FastAPI(title="PaySafe Fraud Scoring", lifespan=lifespan)

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        champion: LoadedChampion | None = getattr(request.app.state, "champion", None)
        if champion is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        return HealthResponse(status="healthy", model_loaded=True)

    @application.post("/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        started = time.perf_counter()
        champion: LoadedChampion | None = getattr(request.app.state, "champion", None)
        if champion is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        logger.info("score_request_received model_version=%s", champion.version)
        try:
            risk_score = score_transaction(champion, payload.model_dump())
        except FeatureContractError as exc:
            logger.warning("score_request_rejected: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("score_prediction_failed model_version=%s", champion.version)
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "score_completed model_version=%s duration_ms=%.2f", champion.version, duration_ms
        )
        return ScoreResponse(risk_score=risk_score, model_version=champion.version)

    return application


def create_app_from_environment() -> FastAPI:
    """Uvicorn factory: each worker resolves configuration and loads its model."""
    return create_app(load_config(force_reload=True))


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the fraud-scoring API")
    parser.add_argument("--config", help="YAML profile; defaults to APP_ENV / CONFIG_PATH")
    args = parser.parse_args()
    config = load_config(args.config, force_reload=True)
    if args.config:
        os.environ["CONFIG_PATH"] = str(Path(args.config).resolve())
    uvicorn.run(
        "fraud_scoring.api:create_app_from_environment",
        factory=True,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        reload=config.server.reload,
        log_level=config.server.log_level.lower(),
    )


if __name__ == "__main__":
    main()
