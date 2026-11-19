import json

from wtms.context.compress import compress_history
from wtms.context.trim import trim_tool_output
from wtms.tokens import count_tokens
from wtms.types import Message, ToolOutput


def test_small_tool_output_unchanged():
    out = ToolOutput("q", '{"rows": [1, 2]}')
    assert trim_tool_output(out) is out


def test_json_arrays_cut_and_marked():
    rows = [{"id": i, "name": f"row {i}", "request_id": "x" * 20} for i in range(200)]
    out = ToolOutput("sql", json.dumps({"rows": rows}))
    trimmed = trim_tool_output(out, max_items=3)
    data = json.loads(trimmed.content)
    assert len(data["rows"]) == 4  # 3 items + omission marker
    assert data["rows"][-1] == "[197 more items omitted]"
    assert "request_id" not in data["rows"][0]
    assert count_tokens(trimmed.content) < count_tokens(out.content)
    assert trimmed.meta["trimmed"] == "true"


def test_log_keeps_errors_and_tail():
    lines = [f"INFO step {i} ok" for i in range(500)]
    lines[100] = "ERROR: deadlock detected"
    out = ToolOutput("logs", "\n".join(lines))
    trimmed = trim_tool_output(out, tail_lines=5).content
    assert "ERROR: deadlock detected" in trimmed
    assert "INFO step 499 ok" in trimmed
    assert "INFO step 50 ok" not in trimmed
    assert "lines omitted]" in trimmed


def test_compress_keeps_recent_and_system():
    hist = [Message("system", "be brief")] + [
        Message("user" if i % 2 == 0 else "assistant", f"Turn {i}. More detail here.") for i in range(10)
    ]
    new, n = compress_history(hist, keep_recent=4)
    assert n == 6
    assert new[0].role == "system"
    assert new[1].content.startswith("Summary of the earlier conversation:")
    assert "Turn 0." in new[1].content and "More detail" not in new[1].content
    assert [m.content for m in new[-4:]] == [m.content for m in hist[-4:]]


def test_compress_noop_when_short():
    hist = [Message("user", "hi"), Message("assistant", "hello")]
    new, n = compress_history(hist, keep_recent=4)
    assert n == 0 and new == hist
