"""Activity-based fallback scanner for Strava segment efforts."""

from .models import ActivityScanResult
from .scanner import ActivityEffortScanner

__all__ = ["ActivityEffortScanner", "ActivityScanResult"]
