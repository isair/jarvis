"""Answer guards for weather grounded through graph, diary or hot memory."""
import re

from evals.helpers import assert_not_fallback_reply


def assert_usable_answer(response: str | None, context: str) -> None:
    """A recorded tool call cannot substitute for an actual reply."""
    assert isinstance(response, str) and response.strip(), f'🗣️ {context}: no answer'
    assert_not_fallback_reply(response, context)


def assert_forecast_reply(response: str | None, forecast: str, context: str) -> None:
    """Require a supplied temperature as well as a usable weather answer."""
    assert_usable_answer(response, context)
    temperatures = re.findall(r'(-?\d+(?:\.\d+)?)°C', forecast)
    assert temperatures, '🌦️ The evaluation fixture must supply temperatures'
    assert any(re.search(r'(?<![\d.])' + re.escape(value) + r'(?![\d.])\s*(?:°\s*C|degrees?\b)', response, re.IGNORECASE)
               for value in temperatures), f'🌦️ {context}: no supplied forecast temperature: {response}'
