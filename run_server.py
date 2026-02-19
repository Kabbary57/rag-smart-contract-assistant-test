#!/usr/bin/env python3
"""
run_server.py – Launch the FastAPI server.

Usage:
    python run_server.py
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
