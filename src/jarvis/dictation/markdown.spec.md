# Markdown Voice Notes

## Overview

When `dictation_markdown_mode` is enabled, the dictation engine converts
spoken structural cues in transcribed text into Markdown syntax before pasting.
This lets a user dictate structured notes hands-free without typing formatting
characters.

## Cue Reference

The formatter matches each entire utterance against the patterns below
(case-insensitive). If the whole utterance is a structural cue, it is replaced.
Otherwise the text is passed through unchanged, so natural-language dictation
is never mangled.

| Spoken cue (example)              | Markdown output            |
|-----------------------------------|----------------------------|
| `new heading: Project Ideas`      | `# Project Ideas`          |
| `heading: Project Ideas`          | `# Project Ideas`          |
| `sub heading: This Week`          | `## This Week`            |
| `bullet point: buy milk`          | `- buy milk`              |
| `bullet: buy milk`                | `- buy milk`              |
| `numbered: first step`            | `1. first step`           |
| `number: first step`              | `1. first step`           |
| `bold: important`                 | `**important**`           |
| `italic: maybe later`             | `*maybe later*`           |
| `code: pip install jarvis`        | `` `pip install jarvis` ``|
| `quote: watch this`               | `> watch this`            |
| `link: Jarvis https://x.io`       | `[Jarvis](https://x.io)`  |
| `new paragraph` / `new line`      | blank-line separator      |
| `divider` / `horizontal rule`     | `---`                     |

The colon after the cue keyword is optional (`heading Ideas` also works).

## Auto-Numbering

When multiple utterances are formatted together via `format_markdown_lines`
(used by callers that batch consecutive dictations), consecutive `numbered`
items are auto-incremented (1., 2., 3., ...) so lists dictated across turns stay
sequential. Any non-numbered utterance resets the counter.

## Design Constraints

- **Stateless per utterance.** No cross-session state, so the user never gets
  a surprise from a "stuck" list mode.
- **Whole-utterance matching only.** Inline cues mid-sentence are deliberately
  ignored to avoid corrupting prose dictation.
- **Fail-open.** If the formatter raises, the raw transcription is returned
  unchanged (see `DictationEngine._format_markdown`).
- **No LLM dependency.** Formatting is pure regex, runs locally, and adds zero
  latency.
