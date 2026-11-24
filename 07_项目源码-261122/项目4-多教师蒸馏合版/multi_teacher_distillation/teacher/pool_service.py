import asyncio
import json
import os
from typing import Any

import ray
from fastapi import FastAPI

from .resident_service import ResidentCoordinator


def create_app() -> FastAPI:
    manifest = json.loads(os.environ["TEACHER_MANIFEST"])
    coordinator = ResidentCoordinator(manifest)
    app = FastAPI()
    app.state.coordinator = coordinator

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "teachers_per_gpu": 2, "layouts": coordinator.layouts()}

    @app.post("/score_batch")
    async def score_batch(payload: dict[str, Any]) -> dict[str, Any]:
        responses = await asyncio.to_thread(coordinator.score_batch, payload["requests"])
        return {"responses": responses}

    return app


app = create_app()
