"""Deterministic identity / capability answers, sourced from a live registry.

Phase 4, Module D. Two responsibilities, both pure and side-effect free:

1. ``build_capability_registry(cfg)`` derives, from the current ``Settings``
   object alone, exactly what Cora can and cannot do right now — which TTS
   engine is active, which speech model runs, whether the Phase-4 memory
   features are enabled, whether any external connector (n8n / MCP) is wired.
   Every status is computed, never invented, so the audit panel and the
   identity answers below tell the truth about the running process.

2. ``answer_identity_question(text, cfg)`` answers reflexive, second-person
   questions ("cine ești", "ce poți face", "ce voce folosești", "ce model
   folosești", "cum funcționează memoria ta", "la ce ești conectată") in
   Romanian, before the LLM, from that same registry. A 4B model asked
   "who are you" invents a persona and a feature list; here the answer is a
   fact the process already holds.

The matchers are deliberately narrow — a false positive would hijack a
question the model should have answered ("spune-mi despre pisici") or, worse,
swallow a memory command ("memorează că..."). A leading-command guard rejects
those outright before any identity pattern is consulted.

Nothing here calls an LLM, touches the network, reads the DB or the live
config, or exposes a secret / token / sensitive absolute path: every rendered
string is passed through :func:`scrub_secrets` as a final structural gate, and
no filesystem path is ever placed into a label, detail, or answer.

The module is inert unless ``cfg.identity_registry_enabled`` is set:
``answer_identity_question`` returns ``None`` when the flag is off, so wiring
it into the local-answer path changes nothing until the owner opts in.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

from ..utils.redact import scrub_secrets

# ------------------------------------------------------------------- statuses

AVAILABLE = "AVAILABLE"          # present and usable, not the active selection
CONFIGURED = "CONFIGURED"        # present, usable AND selected as active
DISABLED = "DISABLED"            # gated off by a config flag (default state)
BLOCKED = "BLOCKED"              # switched on but a prerequisite is missing
NOT_CONNECTED = "NOT_CONNECTED"  # an external connector with nothing wired

STATUSES = frozenset({AVAILABLE, CONFIGURED, DISABLED, BLOCKED, NOT_CONNECTED})


@dataclass(frozen=True)
class Capability:
    """One thing Cora can or cannot do, and why — never invented.

    ``status`` is always one of :data:`STATUSES`. ``CONFIGURED`` implies
    available: it is the strongest positive state (present *and* active).
    """

    key: str
    label: str
    status: str
    detail: str = ""


def _cap(key: str, label: str, status: str, detail: str = "") -> Capability:
    """Construct a Capability, scrubbing the detail as a final safety gate."""
    return Capability(key=key, label=label, status=status,
                      detail=scrub_secrets(detail or "").strip())


def _flag_status(enabled: bool) -> str:
    return AVAILABLE if enabled else DISABLED


def _flag_detail(enabled: bool) -> str:
    return "activ" if enabled else "dezactivat implicit"


# ------------------------------------------------------------------ registry

def build_capability_registry(cfg) -> list[Capability]:
    """Return the deterministic capability list derived from ``cfg``.

    Pure: reads only attributes off the passed ``Settings`` object (via
    ``getattr`` with safe defaults, so a partial object works), performs no
    I/O, and never embeds a filesystem path or secret in any field.
    """
    caps: list[Capability] = []

    tts_engine = str(getattr(cfg, "tts_engine", "piper") or "piper").lower()

    # Primary neural voice — Supertonic F5 when it is the active engine.
    su_voice = str(getattr(cfg, "tts_supertonic_voice", "F5") or "F5")
    su_lang = str(getattr(cfg, "tts_supertonic_language", "ro") or "ro")
    caps.append(_cap(
        "tts_supertonic", "Voce Supertonic (F5)",
        CONFIGURED if tts_engine == "supertonic" else AVAILABLE,
        f"voce {su_voice}, limba {su_lang}",
    ))

    # Piper — automatic fallback voice (or the active engine on default cfg).
    caps.append(_cap(
        "tts_piper", "Voce Piper (rezervă)",
        CONFIGURED if tts_engine == "piper" else AVAILABLE,
        "voce neurală locală, rezervă automată",
    ))

    # Speech recognition — Whisper (large-v3 on the live cfg).
    whisper_model = str(getattr(cfg, "whisper_model", "medium") or "medium")
    caps.append(_cap(
        "asr_whisper", "Recunoaștere vocală Whisper",
        AVAILABLE, f"model {whisper_model}",
    ))

    # Conversation model — local, via Ollama.
    chat_model = str(getattr(cfg, "ollama_chat_model", "") or "").strip()
    caps.append(_cap(
        "chat_model", "Model conversație",
        AVAILABLE,
        f"{chat_model} (local, prin Ollama)" if chat_model else "local, prin Ollama",
    ))

    # External connectors — derived from the MCP map (empty on the live cfg).
    mcps = getattr(cfg, "mcps", {}) or {}
    if not isinstance(mcps, dict):
        mcps = {}
    n8n_present = any("n8n" in str(k).lower() for k in mcps)
    caps.append(_cap(
        "n8n", "Automatizări n8n",
        CONFIGURED if n8n_present else NOT_CONNECTED,
        "conectat" if n8n_present else "neconfigurat",
    ))
    caps.append(_cap(
        "mcp", "Unelte MCP",
        CONFIGURED if mcps else NOT_CONNECTED,
        f"{len(mcps)} unelte conectate" if mcps else "nicio unealtă conectată",
    ))

    # Phase-4 memory / learning features — default OFF ⇒ DISABLED.
    conv_learn = bool(getattr(cfg, "conversation_learning_enabled", False))
    caps.append(_cap("conversation_learning", "Învățare din conversații",
                     _flag_status(conv_learn), _flag_detail(conv_learn)))

    legacy_write = bool(getattr(cfg, "legacy_knowledge_auto_write_enabled", False))
    caps.append(_cap("legacy_knowledge_auto_write",
                     "Scriere automată în graful de cunoștințe",
                     _flag_status(legacy_write), _flag_detail(legacy_write)))

    owner_profile = bool(getattr(cfg, "owner_profile_enabled", False))
    caps.append(_cap("owner_profile", "Profil proprietar",
                     _flag_status(owner_profile), _flag_detail(owner_profile)))

    state_mem = bool(getattr(cfg, "state_memory_enabled", False))
    caps.append(_cap("state_memory", "Memorie de stare (durată)",
                     _flag_status(state_mem), _flag_detail(state_mem)))

    net_learn = bool(getattr(cfg, "internet_learning_enabled", False))
    caps.append(_cap("internet_learning", "Învățare din internet",
                     _flag_status(net_learn), _flag_detail(net_learn)))

    self_eval = bool(getattr(cfg, "self_eval_enabled", False))
    caps.append(_cap("self_eval", "Auto-evaluare",
                     _flag_status(self_eval), _flag_detail(self_eval)))

    audit = bool(getattr(cfg, "audit_panel_enabled", False))
    caps.append(_cap("audit_panel", "Panou de audit",
                     _flag_status(audit), _flag_detail(audit)))

    # Owner-triggered development — DISABLED unless the master switch is on
    # AND a real coding-agent provider is selected. Both on ⇒ CONFIGURED.
    dev_on = bool(getattr(cfg, "owner_triggered_development_enabled", False))
    provider = str(getattr(cfg, "development_agent_provider", "disabled")
                   or "disabled").strip().lower()
    if dev_on and provider != "disabled":
        caps.append(_cap("owner_triggered_development",
                         "Mod dezvoltare (declanșat de proprietar)",
                         CONFIGURED, f"activ, agent {provider}"))
    else:
        caps.append(_cap("owner_triggered_development",
                         "Mod dezvoltare (declanșat de proprietar)",
                         DISABLED, _flag_detail(False)))

    # Premium cloud voice — when switched on it still needs a credential from
    # the Windows Credential Manager, which this pure helper never reads, so
    # the honest status is BLOCKED (gated on a prerequisite), else DISABLED.
    rt_on = bool(getattr(cfg, "openai_realtime_enabled", False))
    caps.append(_cap("premium_voice", "Voce premium OpenAI Realtime",
                     BLOCKED if rt_on else DISABLED,
                     "necesită o cheie în Credential Manager" if rt_on
                     else _flag_detail(False)))

    return caps


# ------------------------------------------------------------- text folding

def _fold(text: str) -> str:
    """Lowercase and strip diacritics — Whisper drops them unpredictably.

    Mirrors ``local_answers._fold`` so the identity matchers below can be
    written as plain ASCII regexes and still match diacritic input.
    """
    t = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


# A leading imperative that is a command, not a question about the self.
# Guarded first so "memorează că...", "spune-mi despre pisici", "notează ..."
# can never be hijacked by an identity pattern that happens to appear later.
_COMMAND_PREFIX = re.compile(
    r"^\s*(memoreaza|memorez|noteaza|retine|tine minte|salveaza|"
    r"spune-?mi despre|zi-?mi despre|povesteste(-?mi)?|"
    r"cauta|deschide|scrie|traduce)\b"
)


# ------------------------------------------------------------- identity regex
# Every pattern runs against folded (lowercased, ASCII) text and is anchored on
# a reflexive / second-person stem so it stays narrow.

_RE_WHO = re.compile(r"\b(cine esti|ce esti tu|cine e cora|tu cine esti)\b")

_RE_CAPABILITIES = re.compile(
    r"\bce poti (?:sa )?fac[ie]\b|\bce stii (?:sa )?faci\b|"
    r"\bcu ce (?:ma |mă )?poti ajuta\b|\bce functii ai\b|"
    r"\bce esti capabila (?:sa faci)?\b"
)

_RE_VOICE = re.compile(
    r"\bce voce (?:ai|folosesti|folositi|ai tu)\b|\bcu ce voce vorbesti\b"
)

_RE_MODEL = re.compile(
    r"\bce model (?:folosesti|ai|utilizezi|ai la baza)\b|"
    r"\bpe ce model (?:functionezi|rulezi|esti)\b|"
    r"\bce (?:llm|ia|inteligenta) folosesti\b"
)

_RE_MEMORY = re.compile(
    r"\bcum functioneaza memoria (?:ta|tale)\b|"
    r"\bcum (?:functioneaza|merge) memoria\b|"
    r"\bcum (?:tii|retii) minte\b|\bcum iti amintesti\b"
)

_RE_CONNECTED = re.compile(
    r"\bla ce (?:sisteme |servicii )?esti conectat\w*\b|"
    r"\bcu ce (?:sisteme |servicii )esti conectat\w*\b|"
    r"\bla ce esti legat\w*\b"
)


# ---------------------------------------------------------------- answer builders

def _voice_answer(cfg) -> str:
    engine = str(getattr(cfg, "tts_engine", "piper") or "piper").lower()
    if engine == "supertonic":
        voice = str(getattr(cfg, "tts_supertonic_voice", "F5") or "F5")
        lang = str(getattr(cfg, "tts_supertonic_language", "ro") or "ro")
        return (
            f"Folosesc vocea neurală Supertonic {voice}, în limba {lang}. "
            "Dacă motorul principal nu răspunde la timp, trec automat pe "
            "vocea Piper."
        )
    if engine == "piper":
        return "Folosesc vocea neurală Piper, generată local pe acest calculator."
    if engine == "chatterbox":
        return "Folosesc vocea neurală Chatterbox, generată local."
    return "Vocea mea este generată local, fără niciun serviciu din cloud."


def _model_answer(cfg) -> str:
    model = str(getattr(cfg, "ollama_chat_model", "") or "").strip()
    if model:
        return (f"Gândesc cu modelul {model}, rulat local prin Ollama, "
                "pe acest calculator.")
    return "Rulez un model de limbaj local prin Ollama, pe acest calculator."


def _who_answer(cfg) -> str:
    return (
        "Sunt Cora, un asistent vocal local care rulează pe acest calculator. "
        "Înțeleg și vorbesc limba română: ascult prin Whisper, gândesc cu un "
        "model rulat local prin Ollama și îți răspund cu voce. Datele și "
        "memoria rămân pe acest PC."
    )


def _memory_answer(cfg) -> str:
    state_on = bool(getattr(cfg, "state_memory_enabled", False))
    learn_on = bool(getattr(cfg, "conversation_learning_enabled", False))
    confirm = bool(getattr(cfg, "memory_require_confirmation", False))
    parts = [
        "Memoria mea are mai multe straturi. În timpul discuției rețin firul "
        "conversației curente, ca să înțeleg contextul."
    ]
    if state_on:
        parts.append(
            "Faptele și preferințele importante le pot păstra ca memorie de "
            "durată, local, pe acest calculator."
        )
    else:
        parts.append(
            "Memoria de durată este momentan dezactivată, așa că, deocamdată, "
            "nu rețin nimic după ce se termină discuția."
        )
    if confirm:
        parts.append("Înainte să rețin ceva ca fapt sigur, îți cer confirmarea.")
    parts.append("Totul rămâne local — nu trimit memoria în cloud.")
    return " ".join(parts)


def _connected_answer(cfg) -> str:
    reg = {c.key: c for c in build_capability_registry(cfg)}
    parts = ["În acest moment rulez local, pe acest calculator."]
    n8n = reg.get("n8n")
    if n8n is not None:
        parts.append(
            "Automatizările n8n nu sunt conectate."
            if n8n.status == NOT_CONNECTED
            else "Sunt conectată la automatizările n8n."
        )
    mcp = reg.get("mcp")
    if mcp is not None:
        parts.append(
            "Nu am unelte externe MCP conectate."
            if mcp.status == NOT_CONNECTED
            else "Am unelte externe MCP conectate."
        )
    parts.append(
        "Recunoașterea vocală, sinteza vocală și modelul de conversație rulează "
        "local, pe acest PC."
    )
    return " ".join(parts)


def _capabilities_answer(cfg) -> str:
    reg = build_capability_registry(cfg)
    active = [c.label for c in reg if c.status in (AVAILABLE, CONFIGURED)]
    disabled = [c.label for c in reg if c.status == DISABLED]
    answer = (
        "Pot să îți spun ora și data, să fac calcule simple și să port o "
        "conversație în română, cu voce."
    )
    if active:
        answer += " Sisteme active acum: " + ", ".join(active) + "."
    if disabled:
        answer += (
            " Unele funcții avansate sunt dezactivate implicit și le pot porni "
            "doar la cererea ta."
        )
    return answer


# (matcher, builder) pairs, tried in order. Distinct stems, so order only
# fixes ties that cannot occur; kept explicit for readability.
_HANDLERS: list[tuple[re.Pattern[str], Callable[[object], str]]] = [
    (_RE_VOICE, _voice_answer),
    (_RE_MODEL, _model_answer),
    (_RE_MEMORY, _memory_answer),
    (_RE_CONNECTED, _connected_answer),
    (_RE_CAPABILITIES, _capabilities_answer),
    (_RE_WHO, _who_answer),
]


def _finalise(answer: Optional[str]) -> Optional[str]:
    """Scrub any secret-shaped token out of the rendered answer, as a net."""
    if not answer:
        return None
    scrubbed = scrub_secrets(answer).strip()
    return scrubbed or None


# ------------------------------------------------------------------- public

def answer_identity_question(text: str, cfg) -> Optional[str]:
    """Return a deterministic Romanian identity answer, or ``None``.

    Inert unless ``cfg.identity_registry_enabled`` is set — returns ``None``
    otherwise, so the local-answer path is unchanged until the owner opts in.
    Returns ``None`` on no match (the question falls through to the LLM) and on
    any leading imperative command, which is never an identity question.

    Args:
        text: the user's query, already stripped of the wake word.
        cfg: the ``Settings`` object (source of both the gate and the facts).
    """
    if not text or not text.strip():
        return None
    if not bool(getattr(cfg, "identity_registry_enabled", False)):
        return None

    folded = _fold(text)
    if _COMMAND_PREFIX.search(folded):
        return None

    for matcher, builder in _HANDLERS:
        if matcher.search(folded):
            return _finalise(builder(cfg))
    return None
