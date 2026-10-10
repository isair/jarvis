# Meal retrieval specification

`fetchMeals` returns meal records and nutrition totals for an inclusive UTC
range. Optional `since_utc` and `until_utc` string fields accept ISO timestamps.
Bounds with Z, an offset or a space separator are converted to the stored UTC
format. Naive bounds are interpreted as UTC, as named by the public fields.

Absent or empty upper bounds resolve to the current UTC instant. Absent or
empty lower bounds resolve to one day before the resolved upper bound.
Invalid timestamps, non-string bounds, non-object arguments and reversed
ranges fail explicitly; they do not produce a successful empty-intake result.

The database stores UTC ISO timestamps with automatic microsecond precision.
The normalised lower bound uses that format, and the upper bound includes an
explicit microsecond fraction so whole-second end instants include stored rows
with or without a zero fraction. Records after the end remain excluded.

The raw result lists each meal ID, description and estimated nutrition, plus
meal count and totals. No model inference occurs in this tool. An empty valid
range returns a successful zero-meal summary. Important range, count and
validation outcomes use nutrition debug logs.
