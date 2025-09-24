import asyncio
import time
import numpy as np
import pytest

from jarvis.listening.streaming_transcriber import StreamingTranscriber, TranscriptionResult

class DummyEngine:
    def __init__(self, **kwargs):
        self.args = type('Args', (), kwargs)

class DummyAudioProcessor:
    def __init__(self, transcription_engine):
        self._lines = [
            {"speaker": 1, "text": "Hello there"},
            {"speaker": 2, "text": "Hi, how are you"},
        ]
        self._created = False
    async def create_tasks(self):
        async def gen():
            # Simulate async arrival
            for line in self._lines:
                await asyncio.sleep(0.01)
                yield {"lines": [line]}
        return gen()
    async def process_audio(self, pcm_bytes):
        return

def test_diarization_two_speakers(monkeypatch):
    received = []

    def on_tr(result: TranscriptionResult):
        received.append(result)

    tr = StreamingTranscriber(
        cfg=type('Cfg', (), {
            'whisperlivekit_backend': 'simulstreaming',
            'whisperlivekit_diarization_enabled': True,
            'whisperlivekit_diarization_backend': 'sortformer',
            'whisperlivekit_frame_threshold': 25,
            'whisperlivekit_beams': 1,
            'whisperlivekit_model': 'tiny'
        })(),
        on_transcript=on_tr,
        enable_microphone=False,
        engine_factory=lambda **kw: DummyEngine(**kw),
        audio_processor_factory=lambda **kw: DummyAudioProcessor(**kw)
    )

    assert tr.start() is True

    # Give the background thread+loop time to process async generator outputs
    time.sleep(0.2)
    tr.stop()

    speakers = {r.speaker_id for r in received}
    texts = [r.text for r in received]

    assert '1' in speakers and '2' in speakers, 'Should receive two distinct speakers'
    assert any('hello' in t.lower() for t in texts)
    assert any('how are you' in t.lower() for t in texts)
