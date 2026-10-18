"""Saved meal descriptions remain literal text in the viewer."""

import json
import re
import subprocess
from html.parser import HTMLParser

import pytest


class MealHeadingParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_heading = False
        self.text = ""
        self.heading_tags = []

    def handle_starttag(self, tag, attrs):
        if self.in_heading:
            self.heading_tags.append(tag)
        if tag == "h3":
            self.in_heading = True

    def handle_endtag(self, tag):
        if tag == "h3":
            self.in_heading = False

    def handle_data(self, data):
        if self.in_heading:
            self.text += data


@pytest.mark.unit
@pytest.mark.parametrize("description", [
    "A meal, café & tea",
    "<b>Literal meal</b>",
    '<img src="invalid" onerror="document.body.dataset.fixtureExecuted=true">',
    '<a href="https://example.invalid">食事</a>',
])
def test_meal_description_is_literal_rendered_text(description):
    from desktop_app.memory_viewer import index

    html = index()
    declarations = list(re.finditer(r"^        (?:async )?function (\w+)\(", html, re.MULTILINE))
    functions = {
        match.group(1): html[match.start():declarations[i + 1].start()]
        for i, match in enumerate(declarations[:-1])
    }
    meal = {"id": 1, "description": description, "ts_utc": "2026-10-12T12:00:00Z"}
    script = "const mealsContent = {innerHTML: '', querySelectorAll: () => []};\n"
    script += functions["escapeHtml"] + functions["renderMeals"]
    script += f"renderMeals({json.dumps([meal])}); console.log(mealsContent.innerHTML);"
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    parsed = MealHeadingParser()
    parsed.feed(result.stdout)
    assert parsed.text == description
    assert parsed.heading_tags == []
