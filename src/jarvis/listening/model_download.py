"""Prepare local Whisper files before GPU model initialisation."""

from jarvis.debug import debug_log


def prepare_mlx_model(repo: str) -> str:
    """Use the Hub cache and native byte progress, without its file-count bar."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class FileCounter(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    print("📥 Checking Whisper model files (first run may download a large model)...", flush=True)
    path = snapshot_download(repo_id=repo, tqdm_class=FileCounter)
    debug_log(f"Whisper model files available: {path}", "voice")
    print("🎤 Loading Whisper into memory and warming up speech recognition...", flush=True)
    return path
