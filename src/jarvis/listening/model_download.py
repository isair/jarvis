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


class ModelDownloadError(RuntimeError):
    """A model preparation failure with a stable recovery category."""

    def __init__(self, category: str, detail: str):
        self.category = category
        super().__init__(f"Whisper model download {category}: {detail}")


def _complete_faster_whisper_model(path: str) -> bool:
    """Require local weights, configuration and tokenisation before loading."""
    from pathlib import Path

    root = Path(path)
    required = ('model.bin', 'config.json', 'tokenizer.json')
    try:
        return (all((root / name).is_file() and (root / name).stat().st_size > 0
                    for name in required)
                and any(file.is_file() and file.stat().st_size > 0
                        for file in root.glob('vocabulary.*')))
    except OSError:
        return False


def _download_error_category(error: Exception) -> str:
    """Retain rate-limit status through the Hub's cache fallback exceptions."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(getattr(current, 'response', None), 'status_code', None) == 429:
            return 'rate_limit'
        current = current.__cause__ or current.__context__
    return 'download'


def _download_worker(connection, model_name: str) -> None:
    """Prepare Hub files in a spawn child without starting audio or the daemon."""
    try:
        from faster_whisper.utils import download_model

        path = download_model(model_name)
        if not _complete_faster_whisper_model(path):
            raise ModelDownloadError('incomplete_download', 'required model files are missing')
        connection.send(('ready', path, ''))
    except Exception as error:
        category = (error.category if isinstance(error, ModelDownloadError)
                    else _download_error_category(error))
        detail = 'HTTP 429' if category == 'rate_limit' else type(error).__name__
        connection.send(('error', category, detail))
    finally:
        connection.close()


def _run_download_worker(model_name: str, timeout_sec: float = 300) -> str:
    """Bound child execution and reap it before exposing its result."""
    import multiprocessing

    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_download_worker, args=(sender, model_name))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(timeout_sec):
            raise ModelDownloadError('timeout', 'download exceeded its time limit')
        try:
            result = receiver.recv()
        except EOFError:
            raise ModelDownloadError('worker_exit', 'download process exited without a result') from None
        process.join(timeout=1)
        if process.is_alive():
            raise ModelDownloadError('worker_exit', 'download process did not stop after its result')
        if process.exitcode != 0:
            raise ModelDownloadError('worker_exit', f'download process exited ({process.exitcode})')
        if not isinstance(result, tuple) or len(result) != 3:
            raise ModelDownloadError('worker_exit', 'invalid download result')
        status, value, detail = result
        if status == 'error':
            raise ModelDownloadError(value, detail)
        if status != 'ready' or not isinstance(value, str) or not _complete_faster_whisper_model(value):
            raise ModelDownloadError('incomplete', 'required model files are missing')
        return value
    except (OSError, RuntimeError) as error:
        if isinstance(error, ModelDownloadError):
            raise
        raise ModelDownloadError('worker_start', type(error).__name__) from error
    finally:
        sender.close()
        receiver.close()
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if not process.is_alive():
                process.close()


def prepare_faster_whisper_model(model_name: str) -> str:
    """Return complete local files, isolating every required network download."""
    from pathlib import Path
    from faster_whisper.utils import download_model

    if Path(model_name).is_dir():
        if _complete_faster_whisper_model(model_name):
            return model_name
        raise ModelDownloadError('incomplete', 'local model directory is missing required files')
    try:
        path = download_model(model_name, local_files_only=True)
        if _complete_faster_whisper_model(path):
            debug_log('Whisper model files available in local cache', 'voice')
            return path
    except Exception as error:
        debug_log(f'Whisper local cache unavailable: {type(error).__name__}', 'voice')
    print('  📥 Checking Whisper model files (first run may download a large model)...', flush=True)
    debug_log('Starting isolated Whisper model download', 'voice')
    try:
        path = _run_download_worker(model_name)
    except ModelDownloadError as error:
        debug_log(f'Whisper download failed: {error.category}', 'voice')
        raise
    debug_log('Whisper model files prepared by isolated download', 'voice')
    return path
