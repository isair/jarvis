"""
Jarvis Voice Assistant Daemon

Main orchestrator that coordinates listening, reply generation, and output.
"""

from __future__ import annotations
import sys
import os
import time
import signal
import threading

# Fix OpenBLAS threading crash in bundled apps (must be before numpy imports)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Fix Windows console encoding for Unicode/emoji characters
# Skip in bundled mode (frozen) - encoding is handled by desktop_app.py
if sys.platform == 'win32' and not getattr(sys, 'frozen', False):
    try:
        import io
        # Only wrap if stdout has a proper binary buffer (not a custom writer)
        if hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'write'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and hasattr(sys.stderr.buffer, 'write'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from typing import Optional
from faster_whisper import WhisperModel

from .config import load_settings
from .memory.db import Database
from .memory.conversation import DialogueMemory, update_diary_from_dialogue_memory
from .output.tts import create_tts_engine
from .tools.registry import initialize_mcp_tools
from .debug import debug_log
from .listening.listener import VoiceListener
from .utils.location import get_location_context, is_location_available

# Global instances for coordination between modules
_global_dialogue_memory: Optional[DialogueMemory] = None
_global_stop_requested: bool = False
_global_tts_engine = None  # TTS engine reference for face animation polling

# Shutdown timeout for diary update (shorter than normal to allow reasonable quit time)
# Desktop app's stop_daemon() should wait at least this long + buffer
SHUTDOWN_DIARY_TIMEOUT_SEC = 45.0

# Callbacks for desktop app to receive diary update progress
# Set by desktop app before calling request_stop()
_diary_update_callbacks: dict = {
    "on_token": None,  # Callable[[str], None] - called for each LLM token
    "on_status": None,  # Callable[[str], None] - called for status updates
    "on_chunks": None,  # Callable[[List[str]], None] - called with pending chunks
    "on_complete": None,  # Callable[[bool], None] - called when done (success/fail)
}


def request_stop() -> None:
    """Request the daemon to stop gracefully. Used by desktop app for QThread shutdown."""
    global _global_stop_requested
    _global_stop_requested = True


def set_diary_update_callbacks(
    on_token=None,
    on_status=None,
    on_chunks=None,
    on_complete=None,
) -> None:
    """
    Set callbacks for diary update progress during shutdown.

    These are used by the desktop app to show a live diary update dialog.

    Args:
        on_token: Called with each LLM token as it's generated
        on_status: Called with status messages
        on_chunks: Called with the list of pending conversation chunks
        on_complete: Called when diary update completes (bool = success)
    """
    global _diary_update_callbacks
    _diary_update_callbacks["on_token"] = on_token
    _diary_update_callbacks["on_status"] = on_status
    _diary_update_callbacks["on_chunks"] = on_chunks
    _diary_update_callbacks["on_complete"] = on_complete


def get_pending_diary_chunks() -> list:
    """Get pending conversation chunks from dialogue memory (for UI display)."""
    global _global_dialogue_memory
    if _global_dialogue_memory is None:
        return []
    return _global_dialogue_memory.get_pending_chunks()


# Diary IPC protocol prefix - desktop app intercepts lines starting with this
DIARY_IPC_PREFIX = "__DIARY__:"


def _emit_diary_event(event_type: str, data) -> None:
    """
    Emit a diary update event to stdout for IPC with desktop app.

    Used in subprocess mode where callbacks aren't available.
    Desktop app intercepts these lines and forwards to diary dialog.

    Args:
        event_type: One of "chunks", "token", "status", "complete"
        data: Event payload (list for chunks, str for token/status, bool for complete)
    """
    import json
    try:
        event = {"type": event_type, "data": data}
        line = f"{DIARY_IPC_PREFIX}{json.dumps(event)}"
        print(line, flush=True)
        # Debug: also print to stderr so we can verify it's being called
        if event_type != "token":  # Don't spam for tokens
            debug_log(f"IPC event emitted: {event_type}", "diary_ipc")
    except Exception as e:
        debug_log(f"IPC emit error: {e}", "diary_ipc")


def is_stop_requested() -> bool:
    """Check if a stop has been requested."""
    return _global_stop_requested


def get_tts_engine():
    """Get the global TTS engine for speaking state polling (used by face widget)."""
    return _global_tts_engine


def _install_signal_handlers() -> None:
    """Ensure signals like Ctrl+Break trigger clean shutdown."""
    def _raise_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _raise_keyboard_interrupt)
            except Exception:
                pass


def _check_and_update_diary(
    db: Database, cfg, verbose: bool = False, force: bool = False, timeout_sec: Optional[float] = None,
    use_callbacks: bool = False, use_ipc: bool = False
) -> None:
    """Check if diary should be updated and perform batch update if needed.

    Args:
        timeout_sec: Optional override for LLM timeout. If None, uses cfg.llm_chat_timeout_sec.
                    During shutdown, a shorter timeout is used to allow graceful quit.
        use_callbacks: If True, uses the global diary update callbacks for UI updates.
        use_ipc: If True, emits diary events to stdout for IPC with desktop app (subprocess mode).
    """
    global _global_dialogue_memory, _diary_update_callbacks

    debug_log(f"diary update check: force={force}, verbose={verbose}", "memory")

    # Helper to safely call callbacks and/or emit IPC events
    def _notify(event_type: str, data):
        # Map event types to callback names
        callback_map = {"chunks": "on_chunks", "status": "on_status", "token": "on_token", "complete": "on_complete"}
        callback_name = callback_map.get(event_type)

        # Call callback if set (bundled mode)
        if use_callbacks and callback_name and _diary_update_callbacks.get(callback_name):
            try:
                _diary_update_callbacks[callback_name](data)
            except Exception:
                pass

        # Emit IPC event (subprocess mode)
        if use_ipc:
            _emit_diary_event(event_type, data)

    if _global_dialogue_memory is None:
        debug_log("diary update skipped: dialogue_memory is None", "memory")
        _notify("complete", False)
        return

    try:
        should_update = force or _global_dialogue_memory.should_update_diary()
        debug_log(f"diary update: should_update={should_update}, force={force}", "memory")

        if should_update:
            pending_chunks = _global_dialogue_memory.get_pending_chunks()
            debug_log(f"diary update: found {len(pending_chunks)} pending chunks", "memory")

            if not pending_chunks:
                debug_log("diary update skipped: no pending chunks", "memory")
                _notify("complete", False)
                return

            # Notify about chunks and status
            _notify("chunks", pending_chunks)
            _notify("status", "Writing diary entry...")

            if verbose:
                try:
                    print("📝 Updating your diary. Please wait… (don't press Ctrl+C again)", file=sys.stderr, flush=True)
                except Exception:
                    pass

            source_app = "stdin" if cfg.use_stdin else "voice"
            effective_timeout = timeout_sec if timeout_sec is not None else cfg.llm_chat_timeout_sec

            # Create token handler that notifies via callback and/or IPC
            # For IPC mode, batch tokens to avoid overwhelming the receiver
            token_buffer = []
            last_flush_time = [time.time()]  # Use list for closure mutability
            TOKEN_FLUSH_INTERVAL = 0.1  # Flush every 100ms

            def on_token_handler(token: str):
                if use_callbacks:
                    # Callbacks can handle individual tokens (same process)
                    _notify("token", token)
                elif use_ipc:
                    # IPC mode: batch tokens to reduce event frequency
                    token_buffer.append(token)
                    now = time.time()
                    if now - last_flush_time[0] >= TOKEN_FLUSH_INTERVAL:
                        if token_buffer:
                            _emit_diary_event("token", "".join(token_buffer))
                            token_buffer.clear()
                        last_flush_time[0] = now

            # Only use token handler if we have callbacks or IPC enabled
            on_token = on_token_handler if (use_callbacks or use_ipc) else None

            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=_global_dialogue_memory,
                ollama_base_url=cfg.ollama_base_url,
                ollama_chat_model=cfg.ollama_chat_model,
                ollama_embed_model=cfg.ollama_embed_model,
                source_app=source_app,
                voice_debug=cfg.voice_debug,
                timeout_sec=effective_timeout,
                force=force,
                on_token=on_token,
            )

            # Flush any remaining tokens in IPC mode
            if use_ipc and token_buffer:
                _emit_diary_event("token", "".join(token_buffer))
                token_buffer.clear()

            if summary_id:
                debug_log(f"diary updated from dialogue memory: id={summary_id}", "memory")
                _notify("complete", True)
            else:
                debug_log("diary update from dialogue memory failed", "memory")
                _notify("complete", False)

            if verbose:
                try:
                    if summary_id:
                        print("✅ Diary update finished.", file=sys.stderr, flush=True)
                    else:
                        print("⚠️ Diary update failed. Shutting down anyway.", file=sys.stderr, flush=True)
                except Exception:
                    pass
        else:
            # No update needed
            _notify("complete", False)
    except Exception as e:
        debug_log(f"diary update check error: {e}", "memory")
        _notify("complete", False)


def main() -> None:
    """Main daemon entry point."""
    global _global_dialogue_memory, _global_stop_requested, _global_tts_engine

    # Reset stop flag at start (in case of restart)
    _global_stop_requested = False

    _install_signal_handlers()

    cfg = load_settings()
    db = Database(cfg.db_path, cfg.sqlite_vss_path)

    debug_log("daemon started", "jarvis")
    print("✓ Daemon started", flush=True)
    print(f"🧠 Using chat model: {cfg.ollama_chat_model}", flush=True)
    print(f"🎤 Using whisper model: {cfg.whisper_model}", flush=True)

    # MCP preflight: discover and cache external MCP tools
    mcps = getattr(cfg, "mcps", {}) or {}
    if mcps:
        print(f"📡 Discovering MCP tools from {len(mcps)} server(s)...", flush=True)
        try:
            mcp_tools = initialize_mcp_tools(mcps, verbose=False)

            # Group tools by server for display
            tools_by_server: dict = {}
            for tool_name in mcp_tools.keys():
                if "__" in tool_name:
                    server_name = tool_name.split("__")[0]
                    if server_name not in tools_by_server:
                        tools_by_server[server_name] = []
                    tools_by_server[server_name].append(tool_name)

            for server_name in mcps.keys():
                count = len(tools_by_server.get(server_name, []))
                if count > 0:
                    print(f"  ✓ {server_name}: {count} tools available", flush=True)
                else:
                    print(f"  ⚠ {server_name}: no tools discovered", flush=True)

            debug_log(f"MCP tools cached: {len(mcp_tools)} total", "mcp")
        except Exception as e:
            debug_log(f"MCP discovery failed: {e}", "mcp")
            print(f"  ⚠ MCP discovery failed: {e}", flush=True)
    else:
        print("📡 No MCP servers configured", flush=True)

    # Initialize dialogue memory with timeout
    print("💾 Initializing dialogue memory...", flush=True)
    _global_dialogue_memory = DialogueMemory(
        inactivity_timeout=cfg.dialogue_memory_timeout,
        max_interactions=20
    )
    print("✓ Dialogue memory initialized", flush=True)

    # Check location detection status
    if cfg.location_enabled:
        location_context = get_location_context(
            config_ip=cfg.location_ip_address,
            auto_detect=cfg.location_auto_detect,
            resolve_cgnat_public_ip=cfg.location_cgnat_resolve_public_ip,
        )
        if location_context == "Location: Unknown":
            print("📍 Location detection not available", flush=True)
            if not is_location_available():
                print("     GeoLite2 database not found. Download from:", flush=True)
                print("     https://www.maxmind.com/en/geolite2/signup", flush=True)
            else:
                print("     Could not detect public IP address.", flush=True)
                print("     Configure 'location_ip_address' in config.json", flush=True)
                print("     or run the setup wizard to configure location.", flush=True)
        else:
            print(f"📍 {location_context}", flush=True)
    else:
        print("📍 Location services disabled", flush=True)

    # Initialize TTS
    print(f"🔊 Initializing TTS engine ({cfg.tts_engine})...", flush=True)
    tts = create_tts_engine(
        engine=cfg.tts_engine,
        enabled=cfg.tts_enabled,
        voice=cfg.tts_voice,
        rate=cfg.tts_rate,
        # Chatterbox parameters
        device=cfg.tts_chatterbox_device,
        audio_prompt_path=cfg.tts_chatterbox_audio_prompt,
        exaggeration=cfg.tts_chatterbox_exaggeration,
        cfg_weight=cfg.tts_chatterbox_cfg_weight,
        # Piper parameters
        piper_model_path=cfg.tts_piper_model_path,
        piper_speaker=cfg.tts_piper_speaker,
        piper_length_scale=cfg.tts_piper_length_scale,
        piper_noise_scale=cfg.tts_piper_noise_scale,
        piper_noise_w=cfg.tts_piper_noise_w,
        piper_sentence_silence=cfg.tts_piper_sentence_silence,
    )
    _global_tts_engine = tts  # Expose for face widget speaking animation
    if tts.enabled:
        tts.start()
        print("✓ TTS engine started", flush=True)
    else:
        print("  TTS disabled", flush=True)

    # Initialize voice listening (only if dependencies available)
    print("🎤 Initializing voice listener (this may take a moment to load Whisper model)...", flush=True)
    voice_thread: Optional[threading.Thread] = None
    voice_thread = VoiceListener(db, cfg, tts, _global_dialogue_memory)
    voice_thread.start()
    print("✓ Voice listener thread started (loading Whisper model in background)", flush=True)
    print("─" * 50, flush=True)

    # Periodic diary update checking
    last_diary_check = time.time()
    diary_check_interval = 60.0

    # Start stdin monitor thread for Windows shutdown signal
    # On Windows, CTRL_BREAK_EVENT doesn't work reliably with CREATE_NO_WINDOW
    # So we also check for stdin being closed as a shutdown signal
    def stdin_monitor():
        global _global_stop_requested
        try:
            # When parent closes our stdin, readline returns empty
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF - stdin closed
                    debug_log("stdin closed, requesting stop", "jarvis")
                    _global_stop_requested = True
                    break
                line = line.strip()
                if line == "SHUTDOWN":
                    debug_log("SHUTDOWN command received, requesting stop", "jarvis")
                    _global_stop_requested = True
                    break
        except Exception:
            pass  # stdin might not be available

    if sys.platform == "win32":
        stdin_thread = threading.Thread(target=stdin_monitor, daemon=True)
        stdin_thread.start()

    try:
        # Main daemon loop
        while not _global_stop_requested:
            time.sleep(1.0)
            now = time.time()

            # Periodically check if diary should be updated
            if now - last_diary_check >= diary_check_interval:
                _check_and_update_diary(db, cfg, verbose=False)
                last_diary_check = now

        # Keep voice thread alive (unless stop requested)
        if voice_thread is not None:
            while voice_thread.is_alive() and not _global_stop_requested:
                time.sleep(0.5)
                _check_and_update_diary(db, cfg, verbose=False)

    except KeyboardInterrupt:
        debug_log("daemon received KeyboardInterrupt", "jarvis")
    finally:
        print("🔄 Daemon shutting down - saving memory...", flush=True)
        debug_log("daemon finally block starting - performing cleanup", "jarvis")

        # Clean shutdown
        if voice_thread is not None:
            debug_log("stopping voice thread...", "jarvis")
            voice_thread.stop()
            try:
                voice_thread.join(timeout=2.0)
            except Exception:
                pass
            debug_log("voice thread stopped", "jarvis")

        # Final diary update before shutdown
        debug_log("performing final diary update (force=True)...", "jarvis")
        print("📝 Updating diary before shutdown...", flush=True)

        # Check dialogue memory status
        if _global_dialogue_memory is None:
            print("⚠️ Dialogue memory is None - nothing to save", flush=True)
        else:
            pending = _global_dialogue_memory.get_pending_chunks()
            print(f"💬 Found {len(pending)} pending conversation chunks", flush=True)

        # Use callbacks if they were set by desktop app (for live UI updates in bundled mode)
        # Use IPC (stdout events) if callbacks not set (subprocess mode)
        use_callbacks = any(_diary_update_callbacks.values())
        use_ipc = not use_callbacks  # Subprocess mode - emit events to stdout
        _check_and_update_diary(db, cfg, verbose=True, force=True, timeout_sec=SHUTDOWN_DIARY_TIMEOUT_SEC, use_callbacks=use_callbacks, use_ipc=use_ipc)
        print("✅ Diary update complete", flush=True)
        debug_log("diary update complete", "jarvis")

        if tts is not None:
            tts.stop()
        db.close()
        debug_log("daemon stopped", "jarvis")
        print("👋 Daemon stopped", flush=True)


if __name__ == "__main__":
    main()
