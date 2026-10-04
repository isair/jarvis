# Local files tool

`localFiles` reads, writes, appends, lists and deletes files locally. It does not call an LLM or a network service. Paths are expanded and resolved before use, and the resolved target must be the user's home directory or a descendant.

- `operation` and `path` are required. Supported operations are `list`, `read`, `write`, `append` and `delete`.
- Write and append require string `content`, which is stored literally as UTF-8. Write creates parent directories.
- Listings use `glob` (default `*`). `recursive` defaults to false and must be a JSON boolean when supplied. Strings, numbers, null, arrays and objects return a correctable tool error without producing a listing.
- False and omitted recursion apply the supplied glob to the target directory. True recursion applies the pattern recursively. Listings show at most 50 sorted entries and report the count of additional entries.
- Reads return at most 10,000 characters, with an explicit truncation marker for longer text. Invalid UTF-8 is replaced.
- Errors return unsuccessful tool results. The reply engine decides how to explain errors or request corrected arguments.
