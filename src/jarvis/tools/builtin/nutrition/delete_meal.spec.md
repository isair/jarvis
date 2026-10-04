# Meal deletion

`deleteMeal` accepts exactly one reference: `id` as a positive integer record ID
or an exact description string, or `meal_description` as an exact description
string. The latter supports the planner's literal concrete-step path without
model resolution. Supplying both reference fields is rejected. ASCII decimal
strings in `id` represent IDs; numeric descriptions use `meal_description`.
Booleans, fractional IDs, non-string descriptions and empty references are rejected.

Description matching is literal and case-sensitive across languages. A single
SQL statement deletes only when exactly one record has that description.
Missing, partial and duplicate descriptions preserve every meal. A failed
deletion asks the caller to use `fetchMeals` and supply a recorded ID.
`fetchMeals` includes record IDs alongside each meal's description and macros.

The tool performs no model inference and never guesses the most recent record.
Queries use bound parameters; failed description writes roll back. Debug logs
record the reference type, deletion outcome and exception type without meal text.
