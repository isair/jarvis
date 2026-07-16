# Screen Awareness

## Overview

Screen awareness lets the assistant see a short, structured snapshot of what the
user is currently looking at, so it can answer questions like "what does this
button do", "summarise what's on my screen", or "what app am I in".

It is **fully opt-in** (`screen_awareness_enabled = false` by default) and
**privacy-first**: every layer degrades to nothing when disabled, and no image
or text ever leaves the machine.

## Layers

| Layer | Config | Behaviour |
|-------|--------|-----------|
| Active window | `screen_awareness_enabled` | Reports the front-most app name + window title. Cheap, no image processing. |
| OCR of visible text | `screen_awareness_ocr` | Screenshots the screen and extracts visible text locally with Tesseract. Requires `pytesseract` + `tesseract`. |

When both are enabled, the context string is:

```
Screen context (what the user is currently looking at):
App: Code.exe
Window: main.py - jarvis
Visible text (OCR): <up to screen_awareness_max_text_chars chars>
```

## Allowlist (privacy gate)

`allowlist_bundles` (already used by the screenshot tool) gates which apps are
captured. An empty allowlist permits every app. Matching is case-insensitive
and ignores `.app` / `.exe` suffixes, so a bundle id like
`com.microsoft.VSCode` matches the process `Code.exe`. When the active app is
not on the allowlist, capture is skipped and `None` is returned.

## Integration

The `ScreenContextProvider` is a lazily-instantiated singleton in
`reply/engine.py`. `_build_enrichment_context_hint` appends the screen context
block to the same hint string already consumed by the memory extractor and the
tool router. That means:

- The tool router can see the screen context and route to `screenshot` /
  `localFiles` / etc. when a query clearly depends on what's on screen.
- The context hint's existing "KNOWN FACTS" framing tells the router a fact
  visible here needs no tool.

The provider caches its snapshot for `cache_seconds` (default 20 s) so
back-to-back queries in one hot window don't each trigger a fresh capture. The
cache is invalidated on new conversation (`reset_screen_provider`).

## Platform Support

- **macOS**: active window via `AppKit.NSWorkspace`; screenshot via
  `screencapture`.
- **Windows**: active window via `win32gui`/`psutil` (fallback `pygetwindow`);
  screenshot via Pillow `ImageGrab`.
- **Linux**: active window via `xdotool`; screenshot via ImageMagick `import`.
  OCR layers gracefully return `None` / `""` when dependencies are missing.

## Constraints

- No network calls. OCR and window lookups are local only.
- Fail-open: any error in capture returns `None` (no context) rather than
  crashing the reply.
- The feature is invisible unless explicitly enabled in config or Settings.
