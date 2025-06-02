"""
FastAPI ASGI application factory and server lifecycle manager.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[CortexAPI] Bootstrapping subsystem connections...")
    yield
    print("[CortexAPI] Releasing network handles and flushing state...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CortexRuntime API",
        version="1.0.0",
        description="High-performance autonomous conversational intelligence runtime API",
        lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
