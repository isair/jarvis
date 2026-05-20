# screenshot tool

Captures the user's display and returns a fenced block for the reply model (OCR + optional local vision).

## Behaviour

- **macOS:** interactive region via `screencapture -i`, then Tesseract OCR when installed.
- **Windows:** primary-display capture via `mss` when installed (thread-safe); otherwise Pillow `ImageGrab` on the Qt main thread only (never `all_screens=True` off the UI thread — that crashes the desktop app). Images are downscaled before OCR/vision.
- **Vision:** when `screen_vision_enabled` is true, sends a JPEG snapshot to `ollama_vision_model` (or `ollama_chat_model` if unset) via Ollama's `images` API.
- **Auto-capture:** when `screen_auto_capture_enabled` is true and `mentions_screen(query)` in `screen_intent.py`, the reply engine captures once before the main loop (no wake word).
- **Other platforms:** fails open with an empty result and debug log (no fabricated demo text).

## Privacy

- Capture runs only when the user asks about the screen, invokes the tool, or uses Pulse → **Skatīt ekrānu** — not continuous monitoring.
- Screen data is fenced as untrusted; the model must not follow on-screen instructions.

## Requirements

- `tesseract` on PATH for OCR text (optional; vision-only still works if the chat model supports images).
- Vision requires an Ollama model that accepts images (e.g. `llava`, `gemma3`, or a multimodal `gemma4` build).
