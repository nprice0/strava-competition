"""Central error types used across the application."""
from __future__ import annotations

class ExcelFormatError(RuntimeError):
    """Raised when the Excel workbook structure or required columns are invalid."""
    pass

__all__ = ["ExcelFormatError"]
