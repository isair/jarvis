# Weather place extraction

`_extract_place_from_user_text` supplies a missing location argument when
`getWeather` has no detected user coordinates. It extracts from the redacted
user utterance using the configured FAST-tier backend, a bounded 1,024-token generation
cap including reasoning and the answer and `llm_tools_timeout_sec`. Blank input, missing configuration, no model,
failed inference and no-place sentinels return `None`.

The first response line is stripped of surrounding quotation/punctuation
wrappers. Internal punctuation belongs to the name, including abbreviated
names such as `St. Petersburg` and `Washington D.C.`. Responses exceeding
60 characters or five words are rejected to limit explanatory output.

A non-empty name is passed as a geocoding query through the existing fixed
geocoding endpoint. Detection/extraction failure requests a city from the user;
the helper does not invent coordinates or produce weather data. Explicit
location arguments and successfully detected coordinates retain precedence.

## Tool-call guidance

The tool description and schema tell the reply model to pass an explicitly
named place in `location`, preserving its geographic qualifiers. That place
takes precedence over the user's detected or remembered location. With no
named place, the model calls with empty arguments and lets the tool resolve
location before asking for clarification. Date and time phrases are not
location arguments.

## Missing personal context

When explicit arguments, detected coordinates and current-utterance extraction
cannot supply a place, the tool returns `missing_context="location"` with its
clarification. The reply layer's shared personal-context resolver can supply
an evidence-backed city and retry. Weather itself does not read diary or graph
memory. See `../../reply/personal_context.spec.md` for the source, freshness,
conflict and attribution contract. Successful detected coordinates and explicit
arguments bypass that resolution.

## Forecast calendar

The raw weather result includes the service's local observation timestamp and
IANA timezone when supplied. Each valid ISO daily date includes its computed
weekday. These labels use the forecast location's calendar, independent of the
host clock, and preserve the recorded date and readings. A malformed daily date
is returned without an invented weekday and does not discard other rows.
