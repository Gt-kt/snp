import pytest

from titan.storage import read_json_object, write_json_atomic


def test_write_json_atomic_round_trips_object(tmp_path):
    path = tmp_path / "state.json"

    write_json_atomic(path, {"a": 1})

    assert read_json_object(path, strict=True) == {"a": 1}


def test_read_json_object_strict_rejects_non_object(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError):
        read_json_object(path, strict=True)


def test_read_json_object_lenient_returns_empty_on_bad_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{bad", encoding="utf-8")

    assert read_json_object(path) == {}
