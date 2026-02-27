#!/usr/bin/env python
"""Simple entry point to run the HolonPolis server."""

import uvicorn

from holonpolis.config import settings

if __name__ == "__main__":
    settings.setup_logging()
    uvicorn.run(
        "holonpolis.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
