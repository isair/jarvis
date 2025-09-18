"""
Jarvis Voice Assistant Daemon

Main orchestrator that coordinates listening, reply generation, and output.
"""

from __future__ import annotations
import time
import signal
import threading
import sounddevice as sd
from typing import Optional
from faster_whisper import WhisperModel

from .config import load_settings
from .memory.db import Database
from .memory.conversation import DialogueMemory, update_diary_from_dialogue_memory
from .output.tts import create_tts_engine
from .tools.external.mcp_client import MCPClient
from .debug import debug_log
from .listening.listener import VoiceListener

# Global instances for coordination between modules
_global_dialogue_memory: Optional[DialogueMemory] = None


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


def _check_and_update_diary(db: Database, cfg, verbose: bool = False, force: bool = False) -> None:
    """Check if diary should be updated and perform batch update if needed."""
    global _global_dialogue_memory
    if _global_dialogue_memory is None:
        return
        
    try:
        should_update = force or _global_dialogue_memory.should_update_diary()
        if should_update:
            pending_chunks = _global_dialogue_memory.get_pending_chunks()
            if not pending_chunks:
                return
            if verbose:
                try:
                    print("📝 Updating your diary. Please wait… (don't press Ctrl+C again)", file=sys.stderr, flush=True)
                except Exception:
                    pass
            
            source_app = "stdin" if cfg.use_stdin else "voice"
            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=_global_dialogue_memory,
                ollama_base_url=cfg.ollama_base_url,
                ollama_chat_model=cfg.ollama_chat_model,
                ollama_embed_model=cfg.ollama_embed_model,
                source_app=source_app,
                voice_debug=cfg.voice_debug,
                timeout_sec=cfg.llm_chat_timeout_sec,
                force=force,
            )
            
            if summary_id:
                debug_log(f"diary updated from dialogue memory: id={summary_id}", "memory")
            else:
                debug_log("diary update from dialogue memory failed", "memory")
                
            if verbose:
                try:
                    if summary_id:
                        print("✅ Diary update finished.", file=sys.stderr, flush=True)
                    else:
                        print("⚠️ Diary update failed. Shutting down anyway.", file=sys.stderr, flush=True)
                except Exception:
                    pass
    except Exception as e:
        debug_log(f"diary update check error: {e}", "memory")


def main() -> None:
    """Main daemon entry point."""
    global _global_dialogue_memory
    
    _install_signal_handlers()
    
    cfg = load_settings()
    db = Database(cfg.db_path, cfg.sqlite_vss_path)
    
    debug_log("daemon started", "jarvis")
    
    # MCP preflight: list available external MCP tools
    mcps = getattr(cfg, "mcps", {}) or {}
    if mcps:
        client = MCPClient(mcps)
        for server_name in mcps.keys():
            try:
                tools = client.list_tools(server_name)
                names = [str(t.get("name")) for t in (tools or []) if t and t.get("name")]
                preview = ", ".join(names[:10])
                debug_log(f"{server_name}: {len(names)} tools available{(': ' + preview) if names else ''}", "mcp")
            except FileNotFoundError as e:
                debug_log(f"{server_name}: command not found – {e}", "mcp")
            except Exception as e:
                debug_log(f"{server_name}: error listing tools: {e}", "mcp")
    
    # Initialize dialogue memory with timeout
    _global_dialogue_memory = DialogueMemory(
        inactivity_timeout=cfg.dialogue_memory_timeout, 
        max_interactions=20
    )
    
    # Initialize TTS
    tts = create_tts_engine(
        engine=cfg.tts_engine,
        enabled=cfg.tts_enabled,
        voice=cfg.tts_voice,
        rate=cfg.tts_rate,
        device=cfg.tts_chatterbox_device,
        audio_prompt_path=cfg.tts_chatterbox_audio_prompt,
        exaggeration=cfg.tts_chatterbox_exaggeration,
        cfg_weight=cfg.tts_chatterbox_cfg_weight
    )
    if tts.enabled:
        tts.start()
    
    # Initialize voice listening (only if dependencies available)
    voice_thread: Optional[threading.Thread] = None
    voice_thread = VoiceListener(db, cfg, tts, _global_dialogue_memory)
    voice_thread.start()
    
    # Periodic diary update checking
    last_diary_check = time.time()
    diary_check_interval = 60.0
    
    try:
        # Main daemon loop
        while True:
            time.sleep(1.0)
            now = time.time()
            
            # Periodically check if diary should be updated
            if now - last_diary_check >= diary_check_interval:
                _check_and_update_diary(db, cfg, verbose=False)
                last_diary_check = now
                
        # Keep voice thread alive
        if voice_thread is not None:
            while voice_thread.is_alive():
                time.sleep(0.5)
                _check_and_update_diary(db, cfg, verbose=False)
                
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        if voice_thread is not None:
            voice_thread.stop()
            try:
                voice_thread.join(timeout=2.0)
            except Exception:
                pass
        
        # Final diary update before shutdown
        _check_and_update_diary(db, cfg, verbose=True, force=True)
        
        if tts is not None:
            tts.stop()
        db.close()
        debug_log("daemon stopped", "jarvis")


if __name__ == "__main__":
    main()
