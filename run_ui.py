#!/usr/bin/env python3
"""
run_ui.py – Launch the Gradio UI (standalone, without FastAPI).

Usage:
    python run_ui.py
    python run_ui.py --port 7861 --share
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Contract Assistant – Gradio UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    from app.ui.gradio_app import launch
    launch(share=args.share, server_port=args.port)
