#!/usr/bin/env python3
"""Convenience runner for the Strava Segment Competition Tool.

Usage:
    python run.py
"""
import logging
from strava_competition.main import main

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    main()
