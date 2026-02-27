"""FastAPI main application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from holonpolis.api.routers import chat, holons
from holonpolis.bootstrap import bootstrap
from holonpolis.config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings.setup_logging()
    logger.info(
        "holonpolis_startup",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        root=str(settings.holonpolis_root),
    )
    bootstrap()
    yield
    # Shutdown
    logger.info("holonpolis_shutdown")


app = FastAPI(
    title="HolonPolis",
    description="HarborPilot2 Evolution Engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return concise request validation details."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": [
                {
                    "field": " -> ".join(str(part) for part in err.get("loc", [])),
                    "type": err.get("type", "unknown"),
                    "msg": err.get("msg", "validation error"),
                }
                for err in exc.errors()
            ],
        },
    )


# Routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(holons.router, prefix="/api/v1")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "root": str(settings.holonpolis_root)}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "HolonPolis",
        "version": "0.1.0",
        "description": "HarborPilot2 Evolution Engine",
    }
