"""Strava Segment Competition package."""

from .main import main
from .models import Runner, Segment, SegmentResult
from .errors import StravaAPIError, ExcelFormatError

__all__ = [
    "main",
    "Runner",
    "Segment",
    "SegmentResult",
    "StravaAPIError",
    "ExcelFormatError",
]
