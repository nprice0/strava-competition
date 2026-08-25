"""Tests for geometry validation projection chunking and coverage semantics."""

from __future__ import annotations

import numpy as np
import pytest

from strava_competition.tools.geometry import validation


def test_chunked_projection_matches_unchunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking the point axis must not change projections or offsets."""
    rng = np.random.default_rng(42)
    polyline = np.cumsum(rng.normal(size=(50, 2)), axis=0)
    points = rng.normal(scale=5.0, size=(257, 2))

    full_proj, full_off = validation._project_onto_polyline(points, polyline)

    monkeypatch.setattr(validation, "_PROJECTION_CHUNK_SIZE", 7)
    chunk_proj, chunk_off = validation._project_onto_polyline(points, polyline)

    np.testing.assert_array_equal(full_proj, chunk_proj)
    np.testing.assert_array_equal(full_off, chunk_off)


def test_chunked_projection_handles_exact_chunk_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point counts that are an exact multiple of the chunk size still work."""
    polyline = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    points = np.array([[float(i), 1.0] for i in range(8)])

    full_proj, full_off = validation._project_onto_polyline(points, polyline)

    monkeypatch.setattr(validation, "_PROJECTION_CHUNK_SIZE", 4)
    chunk_proj, chunk_off = validation._project_onto_polyline(points, polyline)

    np.testing.assert_array_equal(full_proj, chunk_proj)
    np.testing.assert_array_equal(full_off, chunk_off)


def test_compute_coverage_measures_span_not_occupancy() -> None:
    """Documented behaviour: endpoint-only tracks still score a full span."""
    segment = np.array([[0.0, 0.0], [100.0, 0.0]])
    endpoints_only = np.array([[0.0, 0.0], [100.0, 0.0]])

    coverage = validation.compute_coverage(endpoints_only, segment)

    assert coverage.coverage_ratio == pytest.approx(1.0)
    assert coverage.coverage_bounds == pytest.approx((0.0, 100.0))
