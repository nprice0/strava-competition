"""Unit tests for Strava OAuth helper utilities."""

from strava_competition import oauth


def test_mask_token_handles_short_values() -> None:
    assert oauth._mask_token("abcd", visible=4) == "abcd"
    assert oauth._mask_token("abcd", visible=2) == "**cd"
    assert oauth._mask_token("abcd", visible=0) == "****"


def test_mask_token_handles_empty_and_smaller_values() -> None:
    assert oauth._mask_token("", visible=4) == ""
    assert oauth._mask_token("a", visible=4) == "a"
    assert oauth._mask_token("ab", visible=5) == "ab"


def test_mask_token_negative_visible_defaults_to_all_masked() -> None:
    assert oauth._mask_token("abcdef", visible=-2) == "******"
