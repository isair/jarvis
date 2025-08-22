from __future__ import annotations
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Optional
import threading
import difflib

from .config import load_settings
from .redact import redact
from .db import Database
from .embed import get_embedding
from .retrieve import retrieve_top_chunks
from .triggers import evaluate_triggers
from .coach import ask_coach, ask_coach_with_tools
from .profiles import PROFILES, select_profile, select_profile_llm, PROFILE_ALLOWED_TOOLS
from .tts import TextToSpeech
from .tune_player import TunePlayer
from .nutrition import summarize_meals
from .tools import run_tool_with_retries, generate_tools_description, TOOL_SPECS
from .memory import update_daily_conversation_summary, DialogueMemory, update_diary_from_dialogue_memory
from .enhanced_rag import get_enhanced_rag_for_query

try:
    from faster_whisper import WhisperModel  # type: ignore
    import sounddevice as sd  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore
    sd = None  # type: ignore

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None  # type: ignore

import queue
from collections import deque
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

DEFAULT_DEV_PROMPT = PROFILES["developer"].system_prompt

WAKE_WORD = "jarvis"

# Global dialogue memory instance for short-term context
_global_dialogue_memory: Optional[DialogueMemory] = None

# Global voice listener instance for hot window activation
_global_voice_listener: Optional["VoiceListener"] = None


def _chunk_text(text: str, min_chars: int = 500, max_chars: int = 900) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for tok in text.split():
        if size + len(tok) + 1 > max_chars and size >= min_chars:
            parts.append(" ".join(buf))
            buf = []
            size = 0
        buf.append(tok)
        size += len(tok) + 1
    if buf:
        parts.append(" ".join(buf))
    return parts or ([text] if text else [])


def _ingest_iter() -> Iterable[str]:
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        yield s


def _run_coach_on_text(db: Database, cfg, tts: Optional[TextToSpeech], text: str) -> Optional[str]:
    global _global_dialogue_memory
    
    # Start tune immediately when we detect a valid query and begin processing
    tune_player = None
    should_play_tune = cfg.tune_enabled and (tts is None or not tts.is_speaking())
    if should_play_tune:
        tune_player = TunePlayer(enabled=True)
        tune_player.start_tune()
    
    redacted = redact(text)
    chunks = _chunk_text(redacted)
    if chunks:
        ts = datetime.now(timezone.utc).isoformat()
        doc_id, chunk_ids = db.insert_document(
            ts_utc=ts,
            app="stdin" if cfg.use_stdin else "unknown",
            window_title=None,
            url=None,
            sha256_img=None,
            kind="log",
            redaction_ver=1,
            chunks_text=chunks,
        )
        if db.is_vss_enabled:
            for cid, text_chunk in zip(chunk_ids, chunks):
                vec = get_embedding(text_chunk, cfg.ollama_base_url, cfg.ollama_embed_model, timeout_sec=cfg.llm_embedding_timeout_sec)
                if vec is not None:
                    db.upsert_embedding(cid, vec)
    # Try LLM-based routing first; fall back to heuristic
    profile_name = select_profile_llm(cfg.ollama_base_url, cfg.ollama_chat_model, cfg.active_profiles, redacted)
    if cfg.voice_debug:
        try:
            print(f"[debug] selected profile: {profile_name}", file=sys.stderr)
        except Exception:
            pass
    system_prompt = PROFILES.get(profile_name, PROFILES["developer"]).system_prompt
    
    # Get recent dialogue messages for conversation context
    recent_messages = []
    if _global_dialogue_memory and _global_dialogue_memory.has_recent_interactions():
        recent_messages = _global_dialogue_memory.get_recent_messages(max_interactions=5)
    
    # Get enhanced RAG context with LLM-based keyword extraction and multi-strategy retrieval
    enhanced_context = get_enhanced_rag_for_query(
        db=db,
        query=redacted,
        ollama_base_url=cfg.ollama_base_url,
        ollama_embed_model=cfg.ollama_embed_model,
        ollama_chat_model=cfg.ollama_chat_model,
        dialogue_memory=_global_dialogue_memory,
        max_contexts=12,
        include_metadata=cfg.voice_debug,  # Include metadata in debug mode
        voice_debug=cfg.voice_debug
    )
    
    prompt = (
        enhanced_context +
        "\n\nObserved text (redacted excerpt):\n" + redacted[-1200:]
    )
    allowed_tools = PROFILE_ALLOWED_TOOLS.get(profile_name) or list(TOOL_SPECS.keys())
    tools_desc = generate_tools_description(allowed_tools)
    
    if cfg.voice_debug:
        try:
            print(f"[debug] calling LLM: prompt_chars={len(prompt)}, tools_chars={len(tools_desc)}", file=sys.stderr)
        except Exception:
            pass
    
    reply, tool_req, tool_args = ask_coach_with_tools(
        cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, prompt, tools_desc,
        timeout_sec=cfg.llm_tools_timeout_sec, additional_messages=recent_messages, include_location=cfg.location_enabled, 
        config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect
    )
    if cfg.voice_debug:
        try:
            print(f"[debug] LLM returned: has_text={bool(reply)}, tool_req={tool_req}", file=sys.stderr)
        except Exception:
            pass
    if cfg.voice_debug and tool_req:
        try:
            arg_keys = []
            if isinstance(tool_args, dict):
                arg_keys = list(tool_args.keys())
            print(f"[debug] tool requested by model: {tool_req}, args_keys={arg_keys}", file=sys.stderr)
        except Exception:
            pass
    if tool_req:
        result = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name=tool_req,
            tool_args=tool_args,
            system_prompt=system_prompt,
            original_prompt=prompt,
            redacted_text=redacted,
            max_retries=1,
        )
        if result.reply_text:
            # Pass tool results through profile-specific LLM processing
            tool_followup_prompt = (
                f"The user's original query: {redacted}\n\n"
                f"Tool ({tool_req}) returned this information:\n{result.reply_text}\n\n"
                f"Please provide a helpful response to the user based on this information."
            )
            
            # Continue using the existing tune player - don't start/stop a new one
            profile_response = ask_coach(
                cfg.ollama_base_url, 
                cfg.ollama_chat_model, 
                system_prompt, 
                tool_followup_prompt,
                timeout_sec=cfg.llm_chat_timeout_sec,
                additional_messages=recent_messages, 
                include_location=cfg.location_enabled, 
                config_ip=cfg.location_ip_address, 
                auto_detect=cfg.location_auto_detect
            )
            reply = (profile_response or result.reply_text or "").strip()

    # Retry once without tools if model produced no content and didn't request a tool
    if not reply and not tool_req:
        if cfg.voice_debug:
            try:
                print("[debug] retrying without tools...", file=sys.stderr)
            except Exception:
                pass
        
        # Start tune for retry if enabled and not conflicting with TTS
        retry_tune_player = None
        should_play_retry_tune = cfg.tune_enabled and (tts is None or not tts.is_speaking())
        if should_play_retry_tune:
            retry_tune_player = TunePlayer(enabled=True)
            retry_tune_player.start_tune()
        
        try:
            plain = ask_coach(cfg.ollama_base_url, cfg.ollama_chat_model, system_prompt, prompt,
                            timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages, include_location=cfg.location_enabled, 
                            config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect)
            if plain and plain.strip():
                reply = (plain or "").strip()
        finally:
            # Stop retry tune
            if retry_tune_player is not None:
                retry_tune_player.stop_tune()
    # Simple fallback: if user asked about meals/food but model didn't call a tool, fetch today's by default
    if not reply and not tool_req:
        text_l = redacted.lower()
        is_food_mentioned = any(k in text_l for k in [
            "eat", "eaten", "ate", "meal", "meals", "food", "breakfast", "lunch", "dinner"
        ])
        is_list_requested = any(k in text_l for k in [
            "list", "show", "print", "what did i eat", "what i ate", "what have i eaten", "what have i ate"
        ]) or ("what" in text_l and ("eat" in text_l or "ate" in text_l or "eaten" in text_l) and "today" in text_l)
        if is_food_mentioned and is_list_requested:
            if cfg.voice_debug:
                try:
                    print("[debug] fallback: fetching today's meals", file=sys.stderr)
                except Exception:
                    pass
            now = datetime.now(timezone.utc)
            since = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            meals = db.get_meals_between(since, now.isoformat())
            summary = summarize_meals([dict(r) for r in meals])
            follow_sys = "You are a helpful nutrition coach. Turn the following meal summary into a brief, conversational recap with 1-2 suggestions."
            follow_user = summary
            
            # Start tune for nutrition follow-up if enabled and not conflicting with TTS
            nutrition_tune_player = None
            should_play_nutrition_tune = cfg.tune_enabled and (tts is None or not tts.is_speaking())
            if should_play_nutrition_tune:
                nutrition_tune_player = TunePlayer(enabled=True)
                nutrition_tune_player.start_tune()
            
            try:
                follow_text = ask_coach(cfg.ollama_base_url, cfg.ollama_chat_model, follow_sys, follow_user,
                                      timeout_sec=cfg.llm_chat_timeout_sec, additional_messages=recent_messages, include_location=cfg.location_enabled, 
                                      config_ip=cfg.location_ip_address, auto_detect=cfg.location_auto_detect) or ""
                follow_text = (follow_text or "").strip()
                reply = follow_text or summary
            finally:
                # Stop nutrition tune
                if nutrition_tune_player is not None:
                    nutrition_tune_player.stop_tune()
    
    # Intentionally avoid logging the full reply in debug to prevent duplicate output
    if reply:
        # Ensure we never speak or print tool protocol lines
        def _sanitize_out(text_out: str) -> str:
            lines = [ln for ln in (text_out.splitlines() or []) if not ln.strip().upper().startswith("TOOL:")]
            return "\n".join(lines).strip()
        safe_reply = _sanitize_out(reply)
        if safe_reply:
            print(f"\n[jarvis coach:{profile_name}]\n" + safe_reply + "\n", flush=True)
            if tts is not None and tts.enabled:
                # Define completion callback for hot window activation
                def _on_tts_complete():
                    global _global_voice_listener
                    if _global_voice_listener is not None:
                        _global_voice_listener._activate_hot_window()
                
                tts.speak(safe_reply, completion_callback=_on_tts_complete)
    
    # Add interaction to dialogue memory instead of immediately updating diary
    if _global_dialogue_memory is not None:
        try:
            user_text = redacted if redacted and redacted.strip() else ""
            assistant_text = ""
            if reply and reply.strip():
                # Clean the reply for memory (remove tool protocol lines)
                assistant_text = "\n".join([
                    ln for ln in reply.splitlines() 
                    if not ln.strip().upper().startswith("TOOL:")
                ]).strip()
            
            if user_text or assistant_text:
                _global_dialogue_memory.add_interaction(user_text, assistant_text)
                if cfg.voice_debug:
                    try:
                        print(f"[debug] interaction added to dialogue memory", file=sys.stderr)
                    except Exception:
                        pass
        except Exception as e:
            # Don't let memory update failures affect the main conversation flow
            if cfg.voice_debug:
                try:
                    print(f"[debug] dialogue memory error: {e}", file=sys.stderr)
                except Exception:
                    pass
    
    # Always stop tune when processing completes, regardless of code path
    if tune_player is not None:
        tune_player.stop_tune()
    
    return reply


def _commit_buffer(db: Database, cfg, tts: Optional[TextToSpeech], buffer: list[str]) -> None:
    if not buffer:
        return
    text = "\n".join(buffer)
    buffer.clear()
    if not text.strip():
        return
    trig = evaluate_triggers(text)
    if trig.should_fire:
        _run_coach_on_text(db, cfg, tts, text)


class VoiceListener(threading.Thread):
    def __init__(self, db: Database, cfg, tts: Optional[TextToSpeech]) -> None:
        super().__init__(daemon=True)
        global _global_voice_listener
        _global_voice_listener = self
        self.db = db
        self.cfg = cfg
        self.tts = tts
        self.should_stop = False
        self.model = None
        self._audio_q: queue.Queue[Optional["np.ndarray"]] = queue.Queue(maxsize=64)  # type: ignore[name-defined]
        self._pre_roll: deque = deque()
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames: list = []
        self._frame_samples = 0
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        self._vad = None
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None
        # Query assembly state
        self._pending_query: str = ""
        self._is_collecting: bool = False
        self._last_voice_time: float = time.time()
        self._collect_start_time: float = 0.0
        
        # Hot window state - listens without keyword after TTS finishes
        self._is_hot_window_active: bool = False
        self._hot_window_start_time: float = 0.0
        self._last_tts_finish_time: float = 0.0
        # Capture hot window state when voice input starts (before transcription)
        self._was_hot_window_active_at_voice_start: bool = False

    def _is_stop_command(self, text_lower: str) -> bool:
        """Check if the given text contains a stop command that's not part of TTS echo"""
        if not text_lower or not text_lower.strip():
            return False
            
        stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
        stop_sounds = ["sh", "shhh", "shhhh", "ssh", "sssh", "ssssh"]  # Common representations of shushing
        
        # Check if any stop command or sound is present
        detected_commands = []
        
        # Check for stop commands
        for cmd in stop_commands:
            if cmd in text_lower:
                detected_commands.append(cmd)
        
        # Check for shush sounds
        for sound in stop_sounds:
            if sound in text_lower:
                detected_commands.append(sound)
        
        # Check fuzzy matches for short inputs
        if len(text_lower.split()) <= 2:
            try:
                fuzzy_threshold = float(getattr(self.cfg, "stop_command_fuzzy_ratio", 0.8))
                for word in text_lower.split():
                    for cmd in stop_commands:
                        ratio = difflib.SequenceMatcher(a=cmd, b=word).ratio()
                        if ratio >= fuzzy_threshold:
                            detected_commands.append(f"{cmd}~{word}")
            except Exception:
                pass
        
        if not detected_commands:
            return False
        
        # If we found stop commands, check if they're part of current TTS output (echo)
        if self.tts and self.tts.enabled:
            current_tts = (self.tts.get_last_spoken_text() or "").strip().lower()
            if getattr(self.cfg, 'voice_debug', False):
                try:
                    print(f"[debug] echo check: TTS available={bool(current_tts)}, TTS_len={len(current_tts) if current_tts else 0}", file=sys.stderr)
                    if current_tts:
                        print(f"[debug] TTS text: '{current_tts[:100]}...'", file=sys.stderr)
                except Exception:
                    pass
            
            if current_tts and self._is_likely_tts_echo(text_lower, current_tts):
                if getattr(self.cfg, 'voice_debug', False):
                    try:
                        print(f"[debug] stop command ignored as TTS echo: '{text_lower}' (TTS: '{current_tts[:50]}...')", file=sys.stderr)
                    except Exception:
                        pass
                return False
        
        # It's a legitimate stop command (not echo)
        if getattr(self.cfg, 'voice_debug', False):
            try:
                print(f"[debug] legitimate stop command detected: {detected_commands[0]} in '{text_lower}'", file=sys.stderr)
            except Exception:
                pass
        return True
    
    def _is_likely_tts_echo(self, heard_text: str, tts_text: str) -> bool:
        """Check if the heard text is likely an echo of the current TTS output"""
        if not heard_text or not tts_text:
            return False
        
        # Simple similarity check
        similarity = difflib.SequenceMatcher(a=tts_text, b=heard_text).ratio()
        if similarity >= 0.7:  # High similarity suggests echo
            return True
        
        # Check if heard text is a substring of TTS text (partial echo)
        if heard_text in tts_text or tts_text in heard_text:
            return True
        
        # Check if TTS content appears within the heard text (mixed with other audio)
        # This handles cases where heard text is longer than TTS due to background noise or timing
        tts_words = tts_text.split()
        heard_words = heard_text.split()
        
        # Look for consecutive sequences of TTS words in the heard text
        if len(tts_words) >= 3:  # Only check for substantial sequences
            for i in range(len(tts_words) - 2):  # Check 3-word sequences
                sequence = " ".join(tts_words[i:i+3])
                if sequence in heard_text:
                    if getattr(self.cfg, 'voice_debug', False):
                        try:
                            print(f"[debug] TTS sequence found in heard text: '{sequence}'", file=sys.stderr)
                        except Exception:
                            pass
                    return True
        
        # Check if significant portion of TTS words appear in heard text
        if len(tts_words) > 0:
            tts_word_set = set(word.lower().strip('.,!?;:"()[]') for word in tts_words if len(word) > 2)  # Ignore short words
            heard_word_set = set(word.lower().strip('.,!?;:"()[]') for word in heard_words if len(word) > 2)
            
            if len(tts_word_set) > 0:
                overlap_ratio = len(tts_word_set.intersection(heard_word_set)) / len(tts_word_set)
                if overlap_ratio >= 0.6:  # 60% of TTS words found in heard text
                    if getattr(self.cfg, 'voice_debug', False):
                        try:
                            print(f"[debug] High word overlap detected: {overlap_ratio:.2f} ({len(tts_word_set.intersection(heard_word_set))}/{len(tts_word_set)} words)", file=sys.stderr)
                        except Exception:
                            pass
                    return True
        
        return False
    


    def _on_audio(self, indata, frames, time_info, status):  # sounddevice callback
        try:
            if self.should_stop:
                return
            # Always process audio, even during TTS, to allow stop commands
            # Copy to avoid referencing the same buffer
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                pass
        except Exception:
            return
    
    def _activate_hot_window(self) -> None:
        """Activate the hot window for listening without wake word after TTS finishes."""
        if not self.cfg.hot_window_enabled:
            return
        self._is_hot_window_active = True
        self._hot_window_start_time = time.time()
        self._last_tts_finish_time = time.time()
        if self.cfg.voice_debug:
            try:
                print(f"[debug] hot window activated for {self.cfg.hot_window_seconds}s", file=sys.stderr)
            except Exception:
                pass
    
    def _should_expire_hot_window(self) -> bool:
        """Check if hot window should expire due to timeout."""
        if not self._is_hot_window_active:
            return False
        current_time = time.time()
        return (current_time - self._hot_window_start_time) >= self.cfg.hot_window_seconds
    
    def _expire_hot_window(self) -> None:
        """Expire the hot window and stop listening without wake word."""
        if self._is_hot_window_active:
            self._is_hot_window_active = False
            if self.cfg.voice_debug:
                try:
                    print("[debug] hot window expired", file=sys.stderr)
                except Exception:
                    pass
    
    def _reset_hot_window_timer(self) -> None:
        """Reset the hot window timer when voice input starts."""
        # This method is no longer used - hot window timer should not be reset
        # The window should expire based on the original timer from TTS completion
        pass

    def _is_speech_frame(self, frame_f32: "np.ndarray") -> bool:  # type: ignore[name-defined]
        # Fall back to RMS energy gate if no VAD or numpy unavailable
        if np is None:
            return True
        if self._vad is None:
            rms = float(np.sqrt(np.mean(np.square(frame_f32))))
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))
        # Convert float32 [-1,1] to 16-bit PCM bytes as required by webrtcvad
        try:
            pcm16 = np.clip(frame_f32.flatten() * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(pcm16, self._samplerate))
        except Exception:
            return False

    def _filter_noisy_segments(self, segments):
        """Filter out Whisper segments that are likely noise based on confidence scores."""
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.7)
        filtered = []
        
        for seg in segments:
            # Check if segment has confidence attribute (faster_whisper should have this)
            if hasattr(seg, 'avg_logprob'):
                # Convert log probability to confidence-like score (0-1 range)
                # avg_logprob is typically negative, so we transform it
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
                if confidence < min_confidence:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[debug] low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", file=sys.stderr)
                        except Exception:
                            pass
                    continue
            elif hasattr(seg, 'no_speech_prob'):
                # Alternative: use no_speech_prob (higher = more likely to be noise)
                confidence = 1.0 - seg.no_speech_prob
                if confidence < min_confidence:
                    if self.cfg.voice_debug:
                        try:
                            print(f"[debug] low confidence segment filtered: '{seg.text}' (conf: {confidence:.3f})", file=sys.stderr)
                        except Exception:
                            pass
                    continue
            
            filtered.append(seg)
        
        return filtered

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out and should be processed."""
        if not self._is_collecting:
            return
        
        current_time = time.time()
        silence_timeout = current_time - self._last_voice_time >= float(self.cfg.voice_collect_seconds)
        max_timeout = current_time - self._collect_start_time >= float(getattr(self.cfg, "voice_max_collect_seconds", 6.0))
        
        if silence_timeout or max_timeout:
            final_query = self._pending_query.strip() or "what should i do next?"
            if self.cfg.voice_debug:
                try:
                    timeout_type = "silence" if silence_timeout else "max"
                    print(f"[debug] query collected ({timeout_type} timeout): '{final_query}'", file=sys.stderr)
                except Exception:
                    pass
            _run_coach_on_text(self.db, self.cfg, self.tts, final_query)
            self._pending_query = ""
            self._is_collecting = False

    def _finalize_utterance(self) -> None:
        if np is None or not self._utterance_frames:
            # Reset state
            self.is_speech_active = False
            self._silence_frames = 0
            self._utterance_frames = []
            return
        try:
            audio = np.concatenate(self._utterance_frames, axis=0).flatten()
        except Exception:
            audio = None
        
        # Reset state before long-running work
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames = []
        if audio is None or audio.size == 0:  # type: ignore[union-attr]
            return
        
        # Check audio duration to filter out very short segments (likely noise)
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            if self.cfg.voice_debug:
                try:
                    print(f"[debug] audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", file=sys.stderr)
                except Exception:
                    pass
            # Check if hot window should expire even when audio is filtered
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
        
        # Decode with Whisper (no internal VAD since we already segmented)
        try:
            segments, _info = self.model.transcribe(audio, language="en", vad_filter=False)  # type: ignore[union-attr]
            # Filter segments by confidence and apply noise filtering
            filtered_segments = self._filter_noisy_segments(segments)
            text = " ".join(seg.text for seg in filtered_segments).strip()
        except TypeError:
            segments, _info = self.model.transcribe(audio, language="en")  # type: ignore[union-attr]
            # Filter segments by confidence and apply noise filtering
            filtered_segments = self._filter_noisy_segments(segments)
            text = " ".join(seg.text for seg in filtered_segments).strip()
        
        # Basic empty text check
        if not text or not text.strip():
            # Check if hot window should expire even when text is filtered
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
            
        self._handle_transcript(text)

    def _handle_transcript(self, text: str) -> None:
        text_lower = (text or "").strip().lower()
        if not text_lower:
            # Check timeout for any ongoing collection
            if self._is_collecting and (time.time() - self._last_voice_time) >= float(self.cfg.voice_collect_seconds) and self._pending_query.strip():
                _run_coach_on_text(self.db, self.cfg, self.tts, self._pending_query.strip())
                self._pending_query = ""
                self._is_collecting = False
            
            # Check if hot window should expire
            if self._should_expire_hot_window():
                self._expire_hot_window()
            return
        # Priority check for stop commands during TTS - process immediately
        if self.tts is not None and self.tts.enabled and self.tts.is_speaking():
            if self._is_stop_command(text_lower):
                if self.cfg.voice_debug:
                    try:
                        print(f"[voice] stop command detected during TTS: {text_lower}", file=sys.stderr)
                    except Exception:
                        pass
                self.tts.interrupt()
                # Clear any pending audio to make stop more responsive
                try:
                    while not self._audio_q.empty():
                        self._audio_q.get_nowait()
                except Exception:
                    pass
                return
        
        # Guard against echo of our own TTS (but allow stop commands through)
        if self.tts is not None and self.tts.enabled:
            # Always allow stop commands to pass through echo protection
            if not self._is_stop_command(text_lower):
                last_tts = (self.tts.get_last_spoken_text() or "").strip().lower()
                is_same_as_tts = False
                if last_tts and text_lower:
                    ratio = difflib.SequenceMatcher(a=last_tts, b=text_lower).ratio()
                    is_same_as_tts = ratio >= 0.74 or (text_lower in last_tts) or (last_tts in text_lower)
                if is_same_as_tts:
                    return
        if self.cfg.voice_debug and text:
            try:
                print(f"[voice] heard: {text}", file=sys.stderr)
            except Exception:
                pass
        # Check if hot window should expire
        if self._should_expire_hot_window():
            self._expire_hot_window()
        
        # Hot window mode - accept input without wake word
        # Use the captured state from when voice input started (before transcription)
        if self._was_hot_window_active_at_voice_start:
            self._pending_query = (self._pending_query + " " + text_lower).strip()
            self._is_collecting = True
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time
            # Expire hot window after accepting input
            self._expire_hot_window()
            # Reset the captured state
            self._was_hot_window_active_at_voice_start = False
            if self.cfg.voice_debug:
                try:
                    print(f"[debug] hot window input accepted: {text_lower}", file=sys.stderr)
                except Exception:
                    pass
            return
        
        # Wake detection
        wake = getattr(self.cfg, "wake_word", WAKE_WORD)
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake}
        heard_tokens = [t.strip(".,!?;:()[]{}\"'`).-_/") for t in text_lower.split() if t.strip()]
        is_wake = False
        if wake in text_lower:
            is_wake = True
        else:
            try:
                ratio_threshold = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
                for token in heard_tokens:
                    for alias in aliases:
                        if difflib.SequenceMatcher(a=alias, b=token).ratio() >= ratio_threshold:
                            is_wake = True
                            break
                    if is_wake:
                        break
            except Exception:
                is_wake = False
        if is_wake:
            fragment = text_lower
            for alias in aliases:
                fragment = fragment.replace(alias, " ")
            # Clean up punctuation that might be left after wake word removal
            fragment = fragment.strip().lstrip(",.!?;:")
            fragment = fragment.strip()
            if fragment:
                self._pending_query = (self._pending_query + " " + fragment).strip()
            else:
                self._pending_query = (self._pending_query + " what should i do next?").strip()
            self._is_collecting = True
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time
            return
        if self._is_collecting:
            # Accept even single words to avoid dropping short intents
            self._pending_query = (self._pending_query + " " + text_lower).strip()
            self._last_voice_time = time.time()
            # Note: timeout check is now handled in _check_query_timeout() called from audio loop

    def run(self) -> None:
        if WhisperModel is None or sd is None:
            return
        try:
            model_name = getattr(self.cfg, "whisper_model", "small")
            compute = getattr(self.cfg, "whisper_compute_type", "int8")
            self.model = WhisperModel(model_name, device="cpu", compute_type=compute)
        except Exception:
            return
        # Audio and VAD parameters
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        frame_ms = int(getattr(self.cfg, "vad_frame_ms", 20))
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = int(getattr(self.cfg, "vad_pre_roll_ms", 240))
        endpoint_silence_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
        max_utt_ms = int(getattr(self.cfg, "max_utterance_ms", 8000))
        pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
        endpoint_silence_frames = max(1, int(endpoint_silence_ms / frame_ms))
        max_utt_frames = max(1, int(max_utt_ms / frame_ms))

        stream_kwargs = {}
        device_env = (self.cfg.voice_device or '').strip().lower()
        if device_env and device_env not in ("default", "system"):
            try:
                device_index = int(self.cfg.voice_device)  # type: ignore[arg-type]
            except ValueError:
                device_index = None
                try:
                    for idx, dev in enumerate(sd.query_devices()):
                        if isinstance(dev.get("name"), str) and (self.cfg.voice_device or '').lower() in dev.get("name").lower():
                            device_index = idx
                            break
                except Exception:
                    device_index = None
            if device_index is not None:
                stream_kwargs["device"] = device_index
        if self.cfg.voice_debug:
            try:
                if "device" in stream_kwargs:
                    dev = sd.query_devices(stream_kwargs["device"])
                    print(f"[voice] using input device: {dev.get('name')} (index {stream_kwargs['device']})", file=sys.stderr)
                else:
                    print("[voice] using system default input device", file=sys.stderr)
            except Exception:
                pass

        # Open stream
        try:
            stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype="float32",
                blocksize=self._frame_samples,
                callback=self._on_audio,
                **stream_kwargs,
            )
        except Exception:
            return

        with stream:
            while not self.should_stop:
                try:
                    item = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                # Reset marker due to TTS speaking
                if item is None:
                    self.is_speech_active = False
                    self._silence_frames = 0
                    self._utterance_frames = []
                    self._pre_roll.clear()
                    continue
                # Safety
                if np is None:
                    continue
                buf: "np.ndarray" = item  # type: ignore[name-defined]
                # Ensure shape (N, 1)
                try:
                    mono = buf.reshape(-1, buf.shape[-1])[:, 0] if buf.ndim > 1 else buf.flatten()
                except Exception:
                    mono = buf.flatten()
                # Slice into frame-sized chunks
                offset = 0
                total = mono.shape[0]
                while offset + self._frame_samples <= total:
                    frame = mono[offset: offset + self._frame_samples]
                    offset += self._frame_samples
                    # VAD decision
                    is_voice = self._is_speech_frame(frame)
                    if not self.is_speech_active:
                        if is_voice:
                            self.is_speech_active = True
                            
                            # Capture hot window state RIGHT when voice starts (before any delays)
                            self._was_hot_window_active_at_voice_start = self._is_hot_window_active and not self._should_expire_hot_window()
                            if self.cfg.voice_debug and self._was_hot_window_active_at_voice_start:
                                try:
                                    print("[debug] voice input started during active hot window", file=sys.stderr)
                                except Exception:
                                    pass
                            
                            # Seed with pre-roll
                            if self._pre_roll:
                                self._utterance_frames.extend(list(self._pre_roll))
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            # Maintain pre-roll buffer
                            self._pre_roll.append(frame.copy())
                            while len(self._pre_roll) > pre_roll_max_frames:
                                try:
                                    self._pre_roll.popleft()
                                except Exception:
                                    break
                    else:
                        if is_voice:
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            self._silence_frames += 1
                            if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= max_utt_frames:
                                self._finalize_utterance()
                                self._pre_roll.clear()
                        
                        # Check for pending query timeout even when there's no voice activity
                        self._check_query_timeout()
                # Keep any tail smaller than a full frame by pushing it back at next callback via pre-roll
                if offset < total:
                    tail = mono[offset:]
                    if tail.size > 0:
                        self._pre_roll.append(tail.copy())
                        while len(self._pre_roll) > pre_roll_max_frames:
                            try:
                                self._pre_roll.popleft()
                            except Exception:
                                break


def _check_and_update_diary(db: Database, cfg) -> None:
    """Check if diary should be updated and perform batch update if needed."""
    global _global_dialogue_memory
    if _global_dialogue_memory is None:
        return
        
    try:
        if _global_dialogue_memory.should_update_diary():
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
            )
            if cfg.voice_debug:
                try:
                    if summary_id:
                        print(f"[debug] diary updated from dialogue memory: id={summary_id}", file=sys.stderr)
                    else:
                        print("[debug] diary update from dialogue memory failed", file=sys.stderr)
                except Exception:
                    pass
    except Exception as e:
        if cfg.voice_debug:
            try:
                print(f"[debug] diary update check error: {e}", file=sys.stderr)
            except Exception:
                pass


def main() -> None:
    global _global_dialogue_memory
    
    cfg = load_settings()
    db = Database(cfg.db_path, cfg.sqlite_vss_path)

    print("[jarvis] daemon started", file=sys.stderr)
    
    # Initialize dialogue memory with 5-minute inactivity timeout
    _global_dialogue_memory = DialogueMemory(inactivity_timeout=cfg.dialogue_memory_timeout, max_interactions=20)

    tts: Optional[TextToSpeech] = TextToSpeech(enabled=cfg.tts_enabled, voice=cfg.tts_voice, rate=cfg.tts_rate)
    if tts.enabled:
        tts.start()

    voice_thread: Optional[VoiceListener] = None
    if WhisperModel is not None and sd is not None:
        voice_thread = VoiceListener(db, cfg, tts)
        voice_thread.start()

    buffer: list[str] = []
    last_commit = time.time()
    last_diary_check = time.time()
    commit_interval = max(1.0, cfg.capture_interval_sec)
    diary_check_interval = 60.0  # Check for diary updates every minute

    try:
        for snippet in _ingest_iter():
            buffer.append(snippet)
            now = time.time()
            if now - last_commit >= commit_interval:
                _commit_buffer(db, cfg, tts, buffer)
                last_commit = now
            
            # Periodically check if diary should be updated
            if now - last_diary_check >= diary_check_interval:
                _check_and_update_diary(db, cfg)
                last_diary_check = now
        if voice_thread is not None:
            while voice_thread.is_alive():
                time.sleep(0.5)
                # Check for diary updates while waiting
                _check_and_update_diary(db, cfg)
    except KeyboardInterrupt:
        pass
    finally:
        if voice_thread is not None:
            voice_thread.should_stop = True
        _commit_buffer(db, cfg, tts, buffer)
        
        # Final diary update before shutdown to save any pending interactions
        _check_and_update_diary(db, cfg)
        
        if tts is not None:
            tts.stop()
        db.close()
        print("[jarvis] daemon stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
