"""Read the OpenAI API key from Windows Credential Manager only.

Target name: ``Cora.OpenAI``
Username: ``apikey``

Never log, print, or return partial key material to callers beyond
presence checks. On failure return ``None`` so the premium backend can
emit ``MISSING_OPENAI_CREDENTIAL`` and stop.
"""

from __future__ import annotations

from typing import Optional

from ..debug import debug_log

CREDENTIAL_TARGET = "Cora.OpenAI"
CREDENTIAL_USERNAME = "apikey"


class MissingOpenAICredential(Exception):
    """Raised when the Windows credential is absent or unreadable."""

    code = "MISSING_OPENAI_CREDENTIAL"


def read_openai_api_key() -> Optional[str]:
    """Return the API key string, or ``None`` if missing/unreadable.

    The value is never written to logs. Callers must not print it.
    """
    try:
        import win32cred  # type: ignore
    except Exception:
        debug_log("win32cred unavailable — cannot read Cora.OpenAI", "openai")
        return None

    try:
        cred = win32cred.CredRead(CREDENTIAL_TARGET, win32cred.CRED_TYPE_GENERIC)
    except Exception as e:
        # win32cred raises on missing target — treat as absent, never log blob.
        debug_log(
            f"Cora.OpenAI credential read failed: {type(e).__name__}",
            "openai",
        )
        return None

    username = str(cred.get("UserName") or "")
    if username.lower() != CREDENTIAL_USERNAME.lower():
        debug_log(
            "Cora.OpenAI credential username mismatch (expected apikey)",
            "openai",
        )
        return None

    blob = cred.get("CredentialBlob")
    if not blob:
        return None

    key = _decode_credential_blob(blob)
    if not key:
        return None
    return key


def _decode_credential_blob(blob: bytes) -> str:
    """Decode CredentialBlob without logging contents."""
    if isinstance(blob, str):
        return blob.strip()
    # Windows typically stores generic passwords as UTF-16LE.
    for encoding in ("utf-16-le", "utf-8"):
        try:
            text = blob.decode(encoding).rstrip("\x00").strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def require_openai_api_key() -> str:
    """Return the key or raise ``MissingOpenAICredential``.

    Never includes key material in the exception message.
    """
    key = read_openai_api_key()
    if not key:
        raise MissingOpenAICredential(MissingOpenAICredential.code)
    return key
