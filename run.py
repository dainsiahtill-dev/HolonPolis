#!/usr/bin/env python
"""Simple entry point to run the HolonPolis server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "holonpolis.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
