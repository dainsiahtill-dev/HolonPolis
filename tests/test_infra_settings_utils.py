"""Tests for infrastructure env parsing helpers."""

from holonpolis.infrastructure.config.settings_utils import (
    env_bool,
    env_float,
    env_int,
    env_list,
    parse_bool,
    parse_typed_value,
)


def test_parse_bool_handles_common_forms():
    assert parse_bool("true") is True
    assert parse_bool("1") is True
    assert parse_bool("off") is False
    assert parse_bool("0") is False
    assert parse_bool("invalid", default=True) is True


def test_parse_typed_value():
    assert parse_typed_value("42") == 42
    assert parse_typed_value("3.14") == 3.14
    assert parse_typed_value("yes") is True
    assert parse_typed_value("hello") == "hello"


def test_env_helpers(monkeypatch):
    monkeypatch.setenv("HP_TEST_BOOL", "yes")
    monkeypatch.setenv("HP_TEST_INT", "19")
    monkeypatch.setenv("HP_TEST_FLOAT", "2.5")
    monkeypatch.setenv("HP_TEST_LIST", "a, b , ,c")

    assert env_bool("HP_TEST_BOOL", default=False) is True
    assert env_int("HP_TEST_INT", default=1, minimum=5) == 19
    assert env_float("HP_TEST_FLOAT", default=0.1, maximum=5.0) == 2.5
    assert env_list("HP_TEST_LIST", default=["x"]) == ["a", "b", "c"]

