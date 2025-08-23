from __future__ import annotations
import threading
from typing import Optional, Callable
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except Exception:  # pragma: no cover - optional until installed
    FastAPI = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    StaticFiles = None  # type: ignore
    BaseModel = object  # type: ignore
    uvicorn = None  # type: ignore


class ChatRequest(BaseModel):  # type: ignore[misc]
    text: str


class ChatResponse(BaseModel):  # type: ignore[misc]
    text: str


class ApiServer:
    def __init__(self, *, host: str, port: int, cors_origins: list[str], enable_pwa: bool,
                 run_chat: Callable[[str], str], health_check: Optional[Callable[[], bool]] = None) -> None:
        if FastAPI is None:
            raise RuntimeError("FastAPI/uvicorn not installed. Please install server dependencies.")
        self._app = FastAPI(title="Jarvis API")
        self._host = host
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._run_chat = run_chat
        self._health_check = health_check

        # CORS
        if CORSMiddleware is not None:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Routes
        @self._app.get("/health")
        def health() -> dict[str, str]:
            ok = True
            if self._health_check is not None:
                try:
                    ok = bool(self._health_check())
                except Exception:
                    ok = False
            return {"status": "ok" if ok else "error"}

        @self._app.post("/api/chat", response_model=ChatResponse)  # type: ignore[misc]
        def chat(req: ChatRequest) -> ChatResponse:  # type: ignore[misc]
            text = (req.text or "").strip()
            if not text:
                return ChatResponse(text="")
            try:
                reply = str(self._run_chat(text) or "")
            except Exception:
                reply = ""
            return ChatResponse(text=reply)

        if enable_pwa and StaticFiles is not None:
            try:
                web_root = Path(__file__).resolve().parents[1] / "web"
                if web_root.exists():
                    self._app.mount("/", StaticFiles(directory=str(web_root), html=True), name="web")
            except Exception:
                pass

    def start_in_background(self) -> None:
        if uvicorn is None:
            raise RuntimeError("uvicorn not installed. Please install server dependencies.")
        if self._thread is not None:
            return
        def _run() -> None:
            uvicorn.run(self._app, host=self._host, port=self._port, log_level="warning")
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()


# Inline PWA assets removed; served as static files from src/web


