"""Whisper engine capabilities shared by runtime and desktop configuration."""

TURBO_MIN_FASTER_WHISPER_VERSION = "1.1.0"


def is_faster_whisper_turbo_supported() -> bool:
    """Whether the importable faster-whisper engine supports large-v3-turbo."""
    try:
        import faster_whisper
        from packaging.version import Version

        return Version(faster_whisper.__version__) >= Version(TURBO_MIN_FASTER_WHISPER_VERSION)
    except Exception:
        return False
