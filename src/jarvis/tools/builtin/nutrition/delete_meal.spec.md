# Meal deletion

`deleteMeal` accepts exactly one reference: `id` as a positive integer record ID
or an exact description string, or `meal_description` as an exact description
string. The latter supports the planner's literal concrete-step path without
model resolution. Supplying both non-null reference fields is rejected. Null optional fields
are treated as absent. ASCII decimal
strings in `id` represent IDs; numeric descriptions use `meal_description`.
Booleans, fractional IDs, non-string descriptions and empty references are rejected.

Description matching is literal and case-sensitive across languages. A single
SQL statement deletes only when exactly one record has that description.
Missing, partial and duplicate descriptions preserve every meal. A failed
deletion asks the caller to use `fetchMeals` and supply a recorded ID.
`fetchMeals` includes record IDs alongside each meal's description and macros.

For conversational follow-ups, the reply engine supplies recorded resource IDs
to the planner independently of prose summaries. The planner uses the ID of
the referenced meal, including when older meals share its description.

The tool performs no model inference and never guesses the most recent record.
Queries use bound parameters; failed description writes roll back. Debug logs
record the reference type, deletion outcome and exception type without meal text.
