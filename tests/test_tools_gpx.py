"""Tests for the shared GPX helpers and their use by the GPX tools."""

from __future__ import annotations

import pytest

from strava_competition.tools.fetch_activity_gps import streams_to_gpx
from strava_competition.tools.fetch_segment_gpx import segment_to_gpx
from strava_competition.tools.gpx import (
    coerce_coordinates,
    escape_xml,
    format_point_attrs,
)


def test_escape_xml_escapes_all_specials() -> None:
    assert escape_xml("""a&b<c>d"e'f""") == "a&amp;b&lt;c&gt;d&quot;e&apos;f"


def test_coerce_coordinates_accepts_numeric_strings() -> None:
    assert coerce_coordinates("51.5", "-0.12") == (51.5, -0.12)


def test_coerce_coordinates_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="Invalid GPS coordinate pair"):
        coerce_coordinates("abc", 1.0)


def test_format_point_attrs_casts_to_float() -> None:
    assert format_point_attrs("51.5", "-0.12") == 'lat="51.5" lon="-0.12"'


def test_streams_to_gpx_casts_coordinates() -> None:
    streams = {"latlng": [["51.5", "-0.12"]]}
    metadata = {"name": "Test <Run>"}

    gpx = streams_to_gpx(streams, metadata)

    assert '<trkpt lat="51.5" lon="-0.12">' in gpx
    assert "<name>Test &lt;Run&gt;</name>" in gpx


def test_streams_to_gpx_raises_on_non_numeric_coordinates() -> None:
    streams = {"latlng": [["not-a-lat", "-0.12"]]}
    metadata = {"name": "Bad Run"}

    with pytest.raises(ValueError, match="Invalid GPS coordinate pair"):
        streams_to_gpx(streams, metadata)


def test_segment_to_gpx_formats_route_points() -> None:
    segment = {
        "name": "Hill & Dale",
        "points": [(51.5, -0.12), (51.6, -0.13)],
        "distance": 1200.0,
    }

    gpx = segment_to_gpx(segment)

    assert '<rtept lat="51.5" lon="-0.12"/>' in gpx
    assert "<name>Hill &amp; Dale</name>" in gpx
    assert "<rte>" in gpx
