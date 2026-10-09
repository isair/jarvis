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
    """Classify structured failures through the Hub's nested cache errors."""
    import errno
    import ssl

    from requests.exceptions import (
        ConnectionError as RequestConnectionError, RequestException, Timeout,
    )
    from urllib3.exceptions import (
        NewConnectionError, ProtocolError, SSLError as UrllibSSLError,
        TimeoutError as UrllibTimeoutError,
    )

    seen = set()
    errors = []
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        errors.append(current)
        # Requests and urllib3 also store underlying exceptions in args/reason.
        pending.extend(item for item in (
            current.__cause__, current.__context__, getattr(current, 'reason', None),
            *current.args,
        ) if isinstance(item, BaseException))
    if any(getattr(getattr(item, 'response', None), 'status_code', None) == 429
           for item in errors):
        return 'rate_limit'
    filesystem_errors = [item for item in errors if isinstance(item, OSError)
                         and not isinstance(item, (ssl.SSLError, RequestException))]
    disk_errors = {errno.ENOSPC}
    if hasattr(errno, 'EDQUOT'):
        disk_errors.add(errno.EDQUOT)
    if any(item.errno in disk_errors or getattr(item, 'winerror', None) in (39, 112)
           for item in filesystem_errors):
        return 'disk_space'
    if any(item.errno in (errno.EACCES, errno.EPERM, errno.EROFS)
           or getattr(item, 'winerror', None) in (5, 1314)
           for item in filesystem_errors):
        return 'cache_access'
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in errors):
        return 'certificate'
    network_errors = (
        RequestConnectionError, Timeout, NewConnectionError, ProtocolError,
        UrllibSSLError, UrllibTimeoutError, ssl.SSLError, TimeoutError,
    )
    network_codes = {errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
                     errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ECONNABORTED}
    if (any(isinstance(item, network_errors) for item in errors)
            or any(item.errno in network_codes for item in filesystem_errors)):
        return 'network'
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
        if error.category == 'disk_space':
            print('  💡 Free space on the model cache drive, or choose a smaller Whisper model '
                  'in Settings, then restart Jarvis. Any cached files have been kept.', flush=True)
        elif error.category == 'cache_access':
            print('  💡 Access was denied while preparing the speech model. Check cache folder permissions '
                  'and security software, then restart Jarvis. Any cached files have been kept.', flush=True)
        elif error.category == 'certificate':
            print('  💡 The speech model download could not verify the server certificate. '
                  'Check your proxy or security software certificate trust with your administrator, '
                  'then restart Jarvis. Any cached files have been kept.', flush=True)
        elif error.category == 'network':
            print('  💡 The speech model download connection failed. Check your connection, proxy '
                  'and firewall, then restart Jarvis to resume. Any cached files have been kept.', flush=True)
        elif error.category in ('timeout', 'worker_exit'):
            print('  💡 Any cached files have been kept. Check your connection and restart Jarvis '
                  'to resume the speech model download.', flush=True)
            if error.category == 'timeout':
                print('     🎤 For a smaller download, choose a smaller Whisper model in Settings.', flush=True)
            else:
                print('     📋 If the download process stops again, use Report Issue in Logs '
                      'to share what happened.', flush=True)
        raise
    debug_log('Whisper model files prepared by isolated download', 'voice')
    return path
