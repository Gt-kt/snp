import json

from titan.journal import append_journal_event


def test_append_journal_event_writes_jsonl(tmp_path):
    path = tmp_path / "trade_journal.jsonl"

    record = append_journal_event("position_added", {"ticker": "AAPL"}, path=path)

    assert record["event_type"] == "position_added"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["event_type"] == "position_added"
    assert loaded["payload"]["ticker"] == "AAPL"
