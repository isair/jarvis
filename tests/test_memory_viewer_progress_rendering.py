"""Maintenance stream text cannot introduce markup into progress rows."""

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest


class ProgressTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = ""
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text += data


@pytest.mark.unit
@pytest.mark.parametrize("field", ["node", "error", "date"])
@pytest.mark.parametrize("text", ["A record, café & tea", "<b>Literal record</b>", '<img src="invalid" onerror="document.body.dataset.fixtureExecuted=true">'])
def test_maintenance_progress_keeps_streamed_text_literal(field, text):
    from desktop_app.memory_viewer import index

    flow = "consolidate" if field == "node" else "import"
    function = "showConsolidateAllModal" if field == "node" else "showImportDiaryModal"
    progress = {"type": "progress", "processed": 1, "total": 1, "date": "2026-10-12", "facts": 1,
                "node": "A node", "before": 2, "after": 1, "delta": -1}
    progress[field] = text
    if field == "node":
        expected = f"⬇️ {text}: 2 → 1 lines (Δ-1)"
    elif field == "error":
        expected = f"❌ 2026-10-12: error: {text}"
    else:
        expected = f"📅 {text}: 1 fact"
    html = index()
    declarations = list(re.finditer(r"^        (?:async )?function (\w+)\(", html, re.MULTILINE))
    functions = {match.group(1): html[match.start():declarations[i + 1].start()]
                 for i, match in enumerate(declarations[:-1])}
    utilities = html[html.index("        // Shared utilities"):html.index("        // API calls")]
    earlier = {"type": "progress", "processed": 1, "total": 2, "date": "2026-10-11", "facts": 1,
               "node": "Earlier record", "before": 2, "after": 1, "delta": -1}
    earlier_text = "⬇️ Earlier record: 2 → 1 lines (Δ-1)" if field == "node" else "📅 2026-10-11: 1 fact"
    expected = earlier_text + expected
    stream = "\n".join(json.dumps(record) for record in [{"type": "start", "total": 2}, earlier, progress]) + "\n"
    script = Path(__file__).with_name("fixtures").joinpath("viewer_progress_dom.js").read_text(encoding="utf-8")
    script += f"\nconst streamText = {json.dumps(stream)};\n" + utilities + functions[function]
    script += f"\n{function}();\n(async () => {{ await document.getElementById('btn-start-{flow}').handlers.click(); console.log(document.getElementById('{flow}-log').innerHTML); }})();"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=10)
    parsed = ProgressTextParser()
    parsed.feed(result.stdout.strip())
    assert parsed.text == expected
    assert parsed.tags == ["div", "div"]
