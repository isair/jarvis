# Record-contract diagnostics

`opaque_record_cases.py` keeps the metadata-free record fixture and its exact
engine and recovery effect assertions. Invoke it explicitly with the strict
native eval harness. Its failures measure the generic opaque-tool path; they
are reported separately from declared-record qualification and are not passes.

The declared fixture in `test_text_tool_schema.py` supplies the optional canonical
`x-jarvis-records` contract. Both fixture variants retain the same backend effects,
exact call budgets and assertions. The declared-contract suite qualifies the explicit metadata
precondition for record binding.
