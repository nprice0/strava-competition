"""Central error types used across the application."""

from __future__ import annotations


class ExcelFormatError(RuntimeError):
    """Raised when the Excel workbook structure or required columns are invalid."""


class StravaAPIError(RuntimeError):
    """Base error for Strava API failures."""


class StravaPermissionError(StravaAPIError):
    """Raised when the API reports insufficient scopes or authentication issues."""


class StravaPaymentRequiredError(StravaAPIError):
    """Raised when Strava returns HTTP 402 for subscription-only resources."""


class StravaResourceNotFoundError(StravaAPIError):
    """Raised when a segment, activity, or stream does not exist."""


class StravaStreamEmptyError(StravaAPIError):
    """Raised when an activity stream is missing required data series."""


__all__ = [
    "ExcelFormatError",
    "StravaAPIError",
    "StravaPermissionError",
    "StravaPaymentRequiredError",
    "StravaResourceNotFoundError",
    "StravaStreamEmptyError",
]
