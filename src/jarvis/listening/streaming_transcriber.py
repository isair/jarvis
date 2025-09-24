"""
Streaming Transcription Engine using WhisperLiveKit.

Provides real-time speech-to-text with speaker diarization and echo-resistant processing.
"""

from __future__ import annotations
import threading
import time
import platform
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from ..debug import debug_log

if TYPE_CHECKING:
    pass


# WhisperLiveKit imports (mandatory)
from whisperlivekit import TranscriptionEngine, AudioProcessor  # type: ignore
import numpy as np  # type: ignore
import asyncio
import sounddevice as sd
import queue

# MLX Whisper imports (macOS only)
try:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import mlx_whisper
    else:
        mlx_whisper = None
except ImportError:
    mlx_whisper = None


@dataclass
class TranscriptionResult:
    """Result from streaming transcription."""
    text: str
    is_final: bool
    confidence: float
    speaker_id: Optional[str] = None
    timestamp: float = 0.0
    is_partial: bool = False


class StreamingTranscriber:
    """
    Streaming transcription engine using WhisperLiveKit.

    Provides real-time speech-to-text with speaker diarization capabilities.
    """

    def __init__(self, cfg, on_transcript: Callable[[TranscriptionResult], None], *,
                 enable_microphone: bool = True,
                 engine_factory: Optional[Callable[..., TranscriptionEngine]] = None,
                 audio_processor_factory: Optional[Callable[..., AudioProcessor]] = None):
        """
        Initialize streaming transcriber.

        Args:
            cfg: Configuration object
            on_transcript: Callback function for transcription results
        """
        self.cfg = cfg
        self.on_transcript = on_transcript
        self._enable_microphone = enable_microphone
        self._engine_factory = engine_factory or (lambda **kw: TranscriptionEngine(**kw))
        self._audio_processor_factory = audio_processor_factory or (lambda **kw: AudioProcessor(**kw))
        self._should_stop = False
        # Internal engine objects
        self._engine: Optional[TranscriptionEngine] = None
        self._audio_processor: Optional[AudioProcessor] = None

        # Async loop & tasks
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._results_task: Optional[asyncio.Task] = None
        self._audio_consumer_task: Optional[asyncio.Task] = None

        # Audio capture
        self._audio_stream: Optional[sd.InputStream] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._pending_stop = threading.Event()

        # Configuration
        self.backend = getattr(cfg, "whisperlivekit_backend", "simulstreaming")
        self.is_diarization_enabled = getattr(cfg, "whisperlivekit_diarization_enabled", True)
        self.diarization_backend = getattr(cfg, "whisperlivekit_diarization_backend", "sortformer")
        self.frame_threshold = getattr(cfg, "whisperlivekit_frame_threshold", 25)
        self.beams = getattr(cfg, "whisperlivekit_beams", 1)

        # Speaker tracking for enhanced echo detection
        self._known_speakers: set = set()
        self._primary_user_speaker: Optional[str] = None
        self._tts_speaker_id: Optional[str] = None  # Track which speaker might be TTS echo

        # Model selection
        self.model_name = getattr(cfg, "whisperlivekit_model", "medium")

        # Audio device configuration
        self.voice_device = getattr(cfg, "voice_device", None)

        # Auto-detect MLX availability (no config needed - handled by requirements.txt)
        self.has_mlx = self._check_mlx_availability()

        # With mandatory imports we assume availability; any ImportError stops startup earlier.
        self.is_available = True
        self.unavailable_reason = None

    def _check_mlx_availability(self) -> bool:
        """Check if MLX Whisper is available and we're on supported platform."""
        if mlx_whisper is None:
            return False

        # Check if we're on macOS ARM64 (the target platform)
        is_macos_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
        if not is_macos_arm:
            debug_log("MLX Whisper available but not on macOS ARM64 - using standard backend", "streaming")
            return False

        debug_log("MLX Whisper detected and available for acceleration", "streaming")
        return True

    def start(self) -> bool:
        """Start embedded transcription engine and microphone capture."""
        try:
            if self._loop_thread and self._loop and self._loop.is_running():
                return True  # Already started

            self._pending_stop.clear()
            self._audio_queue = asyncio.Queue()

            # Spin up loop in separate thread
            def _run_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                try:
                    self._loop.create_task(self._async_start())
                    self._loop.run_forever()
                finally:
                    pending = asyncio.all_tasks(self._loop)
                    for t in pending:
                        t.cancel()
                    try:
                        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass
                    self._loop.close()

            self._loop_thread = threading.Thread(target=_run_loop, daemon=True)
            self._loop_thread.start()

            # Wait a short moment for loop to initialize engine
            timeout = time.time() + 5
            while (self._engine is None or self._audio_processor is None) and time.time() < timeout:
                time.sleep(0.05)

            # Set up microphone capture (16 kHz mono) if enabled
            if self._enable_microphone:
                self._init_audio_stream()
            debug_log(f"🎤 Embedded streaming transcription started (backend: {self.backend}, model: {self.model_name})", "streaming")
            return True
        except Exception as e:
            debug_log(f"Failed to start embedded streaming transcription: {e}", "streaming")
            return False

    async def _async_start(self) -> None:
        """Async portion of startup executed inside event loop thread."""
        model_options = self._get_model_options()
        # Build transcription engine
        self._engine = self._engine_factory(
            model=self.model_name,
            backend=self.backend,
            diarization=self.is_diarization_enabled,
            frame_threshold=self.frame_threshold,
            beams=self.beams,
            diarization_backend=self.diarization_backend,
            **{k: v for k, v in model_options.items() if k not in {"frame_threshold", "beams", "diarization", "diarization_backend"}}
        )
        # Create audio processor (re-uses singleton engine)
        self._audio_processor = self._audio_processor_factory(transcription_engine=self._engine)

        # Launch tasks
        self._results_task = asyncio.create_task(self._consume_results())
        self._audio_consumer_task = asyncio.create_task(self._consume_audio())

    def _init_audio_stream(self) -> None:
        """Initialize microphone capture stream."""
        if self._audio_stream is not None:
            return
        samplerate = 16000
        channels = 1

        def _callback(indata, frames, time_info, status):  # type: ignore
            if status:
                debug_log(f"audio status: {status}", "streaming")
            if self._audio_queue is None or self._pending_stop.is_set():
                return
            # Convert float32 [-1,1] to int16 PCM bytes
            try:
                pcm = (np.clip(indata[:, 0], -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                # Schedule put into asyncio queue
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, pcm)
            except Exception as e:
                debug_log(f"audio callback error: {e}", "streaming")

        try:
            self._audio_stream = sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                callback=_callback,
                device=self.voice_device or None,
                blocksize=1600,  # ~0.1s
            )
            self._audio_stream.start()
        except Exception as e:
            debug_log(f"Failed to open audio input stream: {e}", "streaming")

    async def _consume_audio(self) -> None:
        """Coroutine to feed audio bytes to audio processor."""
        assert self._audio_queue is not None
        while not self._pending_stop.is_set():
            try:
                pcm_bytes = await self._audio_queue.get()
                if self._audio_processor is not None:
                    await self._audio_processor.process_audio(pcm_bytes)
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_log(f"audio consume error: {e}", "streaming")

    async def _consume_results(self) -> None:
        """Consume results generator from audio processor and emit transcripts."""
        try:
            if self._audio_processor is None:
                return
            results_generator = await self._audio_processor.create_tasks()
            async for response in results_generator:
                lines = response.get("lines") if isinstance(response, dict) else None
                if not lines:
                    continue
                for line in lines:
                    try:
                        text = (line.get("text") or "").strip()
                        if not text:
                            continue
                        speaker = line.get("speaker")
                        speaker_id = str(speaker) if isinstance(speaker, int) and speaker > 0 else None
                        result = TranscriptionResult(
                            text=text,
                            is_final=True,
                            confidence=1.0,
                            speaker_id=speaker_id,
                            timestamp=time.time(),
                            is_partial=False,
                        )
                        self._handle_transcript_internal(result)
                    except Exception as ie:
                        debug_log(f"line parse error: {ie}", "streaming")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            debug_log(f"results consume error: {e}", "streaming")

    def _handle_transcript_internal(self, result: TranscriptionResult) -> None:
        """Internal wrapper to call provided callback safely from loop thread."""
        try:
            self.on_transcript(result)
        except Exception as e:
            debug_log(f"callback error: {e}", "streaming")

    def _get_model_options(self) -> Dict[str, Any]:
        """Get model-specific options."""
        options = {
            "frame_threshold": self.frame_threshold,
            "beams": self.beams,
        }

        # Add diarization if enabled
        if self.is_diarization_enabled:
            options.update({
                "diarization": True,
                "diarization_backend": self.diarization_backend,
            })

        # Use MLX acceleration if available (automatic detection)
        if self.has_mlx:
            options["use_mlx"] = True
            debug_log("🍎 Using MLX Whisper acceleration on macOS", "streaming")

        return options

    # _run_server removed (no external server used now)

    def _handle_transcript(self, transcript_data: Dict[str, Any]) -> None:
        """
        Handle transcript from WhisperLiveKit.

        Args:
            transcript_data: Raw transcript data from WhisperLiveKit
        """
        try:
            # Extract transcript information
            text = transcript_data.get("text", "").strip()
            if not text:
                return

            # Enhanced speaker diarization handling
            speaker_id = transcript_data.get("speaker", None)
            if self.is_diarization_enabled and speaker_id:
                # Log speaker changes for better echo detection
                debug_log(f"🗣️  Speaker {speaker_id}: '{text}'", "diarization")

            # Create transcription result
            result = TranscriptionResult(
                text=text,
                is_final=transcript_data.get("is_final", True),
                confidence=transcript_data.get("confidence", 1.0),
                speaker_id=speaker_id,
                timestamp=time.time(),
                is_partial=not transcript_data.get("is_final", True)
            )

            # Enhanced logging for streaming results
            status_emoji = "✅" if result.is_final else "⏳"
            speaker_info = f" [Speaker {speaker_id}]" if speaker_id else ""
            debug_log(f"{status_emoji} Streaming: '{text}'{speaker_info} (conf: {result.confidence:.2f})", "streaming")

            # Call the callback
            self.on_transcript(result)

        except Exception as e:
            debug_log(f"❌ Error handling transcript: {e}", "streaming")

    def process_audio(self, audio_data: np.ndarray) -> None:
        """Optional external audio injection (bypasses microphone)."""
        if self._loop is None:
            return
        try:
            if audio_data.dtype != np.int16:
                # Convert float32/-1..1 or other numeric to int16
                if np.issubdtype(audio_data.dtype, np.floating):
                    norm = np.clip(audio_data, -1.0, 1.0)
                    pcm_bytes = (norm * 32767).astype(np.int16).tobytes()
                else:
                    pcm_bytes = audio_data.astype(np.int16).tobytes()
            else:
                pcm_bytes = audio_data.tobytes()
            if self._audio_queue is not None:
                self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, pcm_bytes)
        except Exception as e:
            debug_log(f"external process_audio error: {e}", "streaming")

    def stop(self) -> None:
        """Stop the streaming transcription."""
        self._should_stop = True

        self._pending_stop.set()

        # Stop audio stream
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop(); self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        # Shut down asyncio loop
        if self._loop is not None:
            def _stop_loop():
                try:
                    self._loop.stop()
                except Exception:
                    pass
            self._loop.call_soon_threadsafe(_stop_loop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)
        self._loop = None
        self._loop_thread = None
        debug_log("🛑 Embedded streaming transcription stopped", "streaming")

    def is_running(self) -> bool:
        """Check if streaming transcription is running."""
        return self._audio_processor is not None

    def get_capabilities(self) -> Dict[str, bool]:
        """Get transcription capabilities."""
        return {
            "streaming": True,
            "diarization": self.is_diarization_enabled,
            "mlx_acceleration": self.has_mlx,
            "real_time": True,
        }
