"""FastAPI main application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from holonpolis.api.routers import chat, holons
from holonpolis.bootstrap import bootstrap


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    bootstrap()
    yield
    # Shutdown
    # Cleanup if needed


app = FastAPI(
    title="HolonPolis",
    description="HarborPilot2 Evolution Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(holons.router, prefix="/api/v1")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "HolonPolis",
        "version": "0.1.0",
        "description": "HarborPilot2 Evolution Engine",
    }
