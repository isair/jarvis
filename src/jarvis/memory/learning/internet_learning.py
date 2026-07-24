"""Phase 4 · Section F — Controlled internet-learning pipeline.

Policy + candidate-production layer for turning already-fetched web search
results into corroborated, provenance-tagged *fact candidates*. It is a
**pure, offline** transform: this module NEVER opens the network, NEVER
downloads, NEVER runs child processes, NEVER installs packages, and NEVER
executes anything it reads. Search results are DEPENDENCY-INJECTED — the
caller (the web-search tool / integrator) fetches them and hands the raw
``{url, title, text, published_at?}`` dicts to :meth:`ingest_results`. Page
text is treated as UNTRUSTED DATA, never as instructions.

Guarantees (mirroring the rest of the brain-foundation stores):

  * **Default-OFF.** With ``enabled=False`` the pipeline is inert:
    :meth:`plan_research` returns ``None`` and :meth:`ingest_results`
    returns ``[]``. Nothing happens at runtime unless the owner sets
    ``internet_learning_enabled`` AND the feature is wired.
  * **Allowlisted topics only.** Research is refused for any topic that is
    not covered by ``topic_allowlist`` (fail-closed).
  * **Corroboration before belief.** A claim only reaches the corroborated
    tier once ``min_sources`` INDEPENDENT registrable domains assert it.
    Single-source claims are still surfaced (so the caller can decide to
    keep looking) but flagged ``needs_more_sources=True`` and are NOT
    saveable.
  * **web_fact candidates are ALWAYS in the ``candidate`` state.** Promotion
    to a confirmed fact requires the Section-E confirm path (out of scope
    here). :meth:`is_saveable` encodes the fail-closed save gate.
  * **Injection-hardened.** Any result whose text matches the shared
    prompt-injection detector (``safety._INJECTION``) is dropped. Secrets
    are scrubbed from every stored claim.
  * **Fail-open for conversation, fail-closed for saving.** No public method
    raises — a malformed result is skipped, not fatal. On any doubt a claim
    is left non-saveable rather than emitted as a confirmed fact.

Concurrency: **concurrency=1.** All public methods serialize on an internal
lock; the pipeline is designed for single-flight research (one research task
at a time). Rate-limit and budget counters are therefore consistent without
callers needing their own coordination.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .safety import _INJECTION  # shared prompt-injection detector (reused)
from .safety import should_reject_lesson_value  # reused credential/PII gate

try:  # reuse the repo's structural secret scrubber when available
    from ...utils.redact import scrub_secrets as _scrub
except Exception:  # pragma: no cover - defensive
    def _scrub(text: str) -> str:  # type: ignore
        return text

__all__ = [
    "ResearchPlan",
    "WebFactCandidate",
    "InternetLearningPipeline",
]


# ---------------------------------------------------------------------------
# Tokenisation / heuristics (deterministic, language-light).
# ---------------------------------------------------------------------------

# A short, deliberately multilingual stopword set (EN + RO). Kept small on
# purpose: this is a grouping heuristic, not an NLP pipeline.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "and",
    "to", "for", "with", "at", "by", "as", "it", "its", "this", "that", "from",
    "be", "has", "have", "had", "will", "about", "into", "than", "then",
    "este", "sunt", "era", "erau", "un", "o", "si", "și", "la", "in", "în",
    "de", "cu", "pe", "ca", "care", "sau", "dar", "acest", "aceasta", "din",
})

# Negation markers (EN + RO) — a polarity flip contributes to the value
# signature so "X is safe" and "X is not safe" become conflicting claims.
_NEG = frozenset({
    "not", "no", "never", "cannot", "without",
    "nu", "niciodata", "niciodată", "nicio", "fara", "fără", "nici",
})

# Temporal markers → the fact is time-bound and must carry an expiry. Any
# explicit "now/today/current/latest" phrase, or a 19xx/20xx year, qualifies.
_TEMPORAL = re.compile(
    r"(?i)\b("
    r"today|now|current(?:ly)?|latest|recent(?:ly)?|as\s+of|"
    r"this\s+(?:year|month|week)|"
    r"azi|acum|curent[aă]?|prezent|recent[aă]?|"
    r"cel\s+mai\s+recent"
    r")\b|\b(?:19|20)\d{2}\b"
)

_WORD_RE = re.compile(r"[0-9a-zăâîșțáéíóúäöü]+", re.UNICODE)
# A number plus an OPTIONAL immediately-following short unit token. Long unit
# words (meters/feet) are already content tokens that separate the subject_key,
# so only short symbols (m, ft, km, kg …) need to enter the value signature to
# stop "300 m" and "300 ft" from colliding on a bare "300". The trailing \b
# keeps the unit whole, so it never captures a prefix of a longer word.
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*([a-zăâîșțáéíóúäöü]{1,3})?\b",
    re.UNICODE,
)

# Registrable-domain override for the handful of common two-level public
# suffixes; everything else falls back to "last two labels".
_TWO_LEVEL_TLDS = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "com.au", "co.nz",
    "com.br", "co.in", "org.au", "gov.au",
})

_TRUSTED_DOMAINS = frozenset({
    "wikipedia.org", "britannica.com", "nature.com", "science.org",
    "who.int", "nih.gov", "nasa.gov", "esa.int", "reuters.com",
})
_TRUSTED_SUFFIXES = (".gov", ".edu", ".mil", ".int")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _content_tokens(text: str) -> List[str]:
    # Negation markers are removed here so the affirmative and negated forms of
    # a claim ("X is safe" / "X is not safe") share the SAME subject_key and
    # differ only in _value_sig's polarity flag — that is what lets
    # _flag_contradictions pair them and hold the affirmative fail-closed.
    return [
        t for t in _tokens(text)
        if len(t) >= 3 and not t.isdigit()
        and t not in _STOPWORDS and t not in _NEG
    ]


def _subject_key(text: str) -> str:
    """A stable, order-independent signature of a claim's *subject* — the
    content words with numbers and stopwords removed. Two phrasings of the
    same claim collapse to the same key; conflicting values differ only in
    the value signature, not here."""
    toks = sorted(set(_content_tokens(text)))
    if toks:
        return " ".join(toks[:12])
    # No content words (all numbers/stopwords): fall back to the normalized
    # raw text so identical claims still group but unrelated ones do not.
    return "raw:" + " ".join(_tokens(text))[:60]


def _value_sig(text: str) -> str:
    """Signature of a claim's asserted *value*: its set of numbers (each with
    the short unit that immediately follows it, so "300 m" and "300 ft" do not
    collide on "300") plus a polarity flag. Same subject + different value
    signature == contradiction.

    NOTE: an *empty* value signature (a qualitative, number-free claim) is
    still treated as corroboratable across independent domains — that is the
    established contract (see test_two_independent_domains_...). Making the
    empty signature non-corroboratable would break that contract, so it is
    intentionally left as-is; only the unit-collision half of the review's
    optional item is implemented here."""
    parts = []
    for m in _NUM_UNIT_RE.finditer(text or ""):
        num = m.group(1)
        unit = (m.group(2) or "").lower()
        # Never fold a stopword or negation marker in as a "unit" (e.g. the
        # year in "2024 the …"); those carry no value information.
        if unit in _STOPWORDS or unit in _NEG:
            unit = ""
        parts.append(num + unit)
    neg = any(t in _NEG for t in _tokens(text))
    return "|".join(sorted(parts)) + ("|NEG" if neg else "")


def _registrable_domain(url: str) -> str:
    """Best-effort registrable domain for independence checks. Pure string
    parsing — no DNS, no network."""
    try:
        host = (urlparse(url).hostname or "").lower().strip()
    except Exception:
        host = ""
    # Normalize the FQDN: drop the optional root dot and any empty labels so
    # "example.com" and "example.com." resolve to the SAME registrable domain
    # instead of masquerading as two independent sources.
    host = host.rstrip(".")
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    labels = [lbl for lbl in host.split(".") if lbl]
    if len(labels) < 2:
        # A bare/single-label host is not a usable registrable domain.
        return ""
    if len(labels) == 2:
        return ".".join(labels)
    last_two = ".".join(labels[-2:])
    if last_two in _TWO_LEVEL_TLDS and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def _trust_for(domain: str) -> str:
    if not domain:
        return "unknown"
    if domain in _TRUSTED_DOMAINS or domain.endswith(_TRUSTED_SUFFIXES):
        return "high"
    return "unverified"


def _is_temporal(text: str) -> bool:
    return bool(_TEMPORAL.search(text or ""))


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------

@dataclass
class ResearchPlan:
    """The (offline) plan produced for an allowed topic. Carries no results —
    it only records *what* a search would look for and the policy envelope."""

    topic: str
    normalized_topic: str
    min_sources: int
    max_results: int
    rate_limit_per_min: int
    budget_remaining: int
    queries: List[str]
    created_at: str
    notes: str = ""


@dataclass
class WebFactCandidate:
    """A proposed web-derived fact. NEVER a confirmed fact — ``state`` is
    always ``"candidate"``; promotion is the Section-E confirm path's job."""

    topic: str
    subject_key: str
    value_sig: str
    claim_text: str
    state: str = "candidate"  # invariant: web facts are never 'confirmed' here
    source_count: int = 1  # number of INDEPENDENT registrable domains
    needs_more_sources: bool = True
    contradicted: bool = False
    conflicting_values: List[str] = field(default_factory=list)
    is_temporal: bool = False
    expires_at: Optional[str] = None
    provenance: List[dict] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "subject_key": self.subject_key,
            "value_sig": self.value_sig,
            "claim_text": self.claim_text,
            "state": self.state,
            "source_count": self.source_count,
            "needs_more_sources": self.needs_more_sources,
            "contradicted": self.contradicted,
            "conflicting_values": list(self.conflicting_values),
            "is_temporal": self.is_temporal,
            "expires_at": self.expires_at,
            "provenance": [dict(p) for p in self.provenance],
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Pipeline.
# ---------------------------------------------------------------------------

class InternetLearningPipeline:
    """Policy + candidate-production for controlled internet learning.

    All state-mutating public methods serialize on ``self._lock`` (concurrency
    == 1). The pipeline holds no network handles and performs no I/O.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        topic_allowlist,
        min_sources: int = 2,
        max_results: int = 5,
        rate_limit_per_min: int = 30,
        budget: int = 1000,
        temporal_ttl_hours: int = 24,
    ) -> None:
        self.enabled = bool(enabled)
        # Normalize the allowlist once: lowercased, stripped, de-duped, no blanks.
        self.topic_allowlist = [
            e.strip().lower()
            for e in (topic_allowlist or [])
            if isinstance(e, str) and e.strip()
        ]
        # Hard internal floor of 2: a single source is NEVER enough to confirm
        # a web fact, even if the caller/config passes min_sources=1.
        self.min_sources = max(2, int(min_sources))
        self.max_results = max(1, int(max_results))
        self.rate_limit_per_min = max(1, int(rate_limit_per_min))
        self._budget_remaining = max(0, int(budget))
        self._temporal_ttl = timedelta(hours=max(1, int(temporal_ttl_hours)))
        self._lock = threading.Lock()
        self._plan_times: "deque[datetime]" = deque()

    # -- policy ------------------------------------------------------------
    @property
    def budget_remaining(self) -> int:
        return self._budget_remaining

    def _topic_allowed(self, topic: str) -> bool:
        if not self.topic_allowlist:
            return False
        norm = " ".join(_tokens(topic))
        if not norm:
            return False
        norm_tokens = set(norm.split())
        for entry in self.topic_allowlist:
            # Whole-token match, or the allowlist phrase as a substring of the
            # normalized topic (so "python" allows "python packaging").
            if entry in norm_tokens or entry in norm:
                return True
        return False

    def plan_research(self, topic: str) -> Optional[ResearchPlan]:
        """Return a :class:`ResearchPlan` for an allowed topic, else ``None``.

        Returns ``None`` when the pipeline is disabled, the topic is not
        covered by the allowlist, the topic is empty, the per-minute rate
        limit is exhausted, or the budget is spent. Never raises."""
        try:
            with self._lock:
                if not self.enabled:
                    return None
                topic = (topic or "").strip()
                if not topic or not self._topic_allowed(topic):
                    return None
                if self._budget_remaining <= 0:
                    return None
                now = _utc_now()
                # Sliding 60s window rate limit.
                cutoff = now - timedelta(seconds=60)
                while self._plan_times and self._plan_times[0] < cutoff:
                    self._plan_times.popleft()
                if len(self._plan_times) >= self.rate_limit_per_min:
                    return None
                self._plan_times.append(now)
                norm = " ".join(_tokens(topic))
                queries = [topic]
                # Purely string-derived query variants; NO network is touched.
                if not _TEMPORAL.search(topic):
                    queries.append(f"{topic} overview")
                queries.append(f"{topic} latest")
                return ResearchPlan(
                    topic=topic,
                    normalized_topic=norm,
                    min_sources=self.min_sources,
                    max_results=self.max_results,
                    rate_limit_per_min=self.rate_limit_per_min,
                    budget_remaining=self._budget_remaining,
                    queries=queries,
                    created_at=_iso(now),
                    notes="offline plan; results must be fetched by the caller",
                )
        except Exception:
            # Fail-open for the conversation path: never propagate.
            return None

    # -- ingestion ---------------------------------------------------------
    def ingest_results(self, topic: str, results) -> List[WebFactCandidate]:
        """Turn already-fetched search ``results`` into fact candidates.

        Each result is ``{url, title, text, published_at?}``. Page ``text`` is
        UNTRUSTED — it is never treated as instructions. Results matching the
        prompt-injection detector are dropped; secrets are scrubbed; claims are
        grouped by (subject, value); corroboration and contradictions are
        computed across INDEPENDENT registrable domains. Returns ``[]`` when
        disabled, the topic is not allowed, or on any error (fail-open). Every
        returned candidate is in the ``candidate`` state — never confirmed."""
        try:
            with self._lock:
                if not self.enabled:
                    return []
                topic = (topic or "").strip()
                if not topic or not self._topic_allowed(topic):
                    return []
                if not results or self._budget_remaining <= 0:
                    return []

                fetched_at = _iso(_utc_now())
                capped = list(results)[: self.max_results]
                # Charge the budget for the results actually considered.
                self._budget_remaining = max(
                    0, self._budget_remaining - len(capped)
                )

                # groups: claim_key -> aggregated claim state.
                groups: Dict[tuple, dict] = {}
                for res in capped:
                    parsed = self._parse_result(res)
                    if parsed is None:
                        continue
                    key = (parsed["subject_key"], parsed["value_sig"])
                    grp = groups.get(key)
                    if grp is None:
                        grp = {
                            "subject_key": parsed["subject_key"],
                            "value_sig": parsed["value_sig"],
                            "claim_text": parsed["claim_text"],
                            "is_temporal": parsed["is_temporal"],
                            "domains": set(),
                            "provenance": [],
                        }
                        groups[key] = grp
                    grp["is_temporal"] = grp["is_temporal"] or parsed["is_temporal"]
                    # Prefer the longest scrubbed claim text as the representative.
                    if len(parsed["claim_text"]) > len(grp["claim_text"]):
                        grp["claim_text"] = parsed["claim_text"]
                    grp["domains"].add(parsed["domain"])
                    grp["provenance"].append({
                        "source_url": parsed["url"],
                        "title": parsed["title"],
                        "fetched_at": fetched_at,
                        "trust": _trust_for(parsed["domain"]),
                        "published_at": parsed["published_at"],
                    })

                candidates = self._build_candidates(topic, groups, fetched_at)
                self._flag_contradictions(candidates)
                return candidates
        except Exception:
            # Fail-open for the conversation path: never propagate.
            return []

    # -- helpers -----------------------------------------------------------
    def _parse_result(self, res) -> Optional[dict]:
        """Validate + normalize a single result dict. Returns ``None`` to skip
        (missing fields, injection-shaped text, or empty after scrubbing)."""
        if not isinstance(res, dict):
            return None
        url = str(res.get("url") or "").strip()
        text = str(res.get("text") or "").strip()
        title = str(res.get("title") or "").strip()
        published_at = res.get("published_at")
        if not url or not text:
            return None
        # UNTRUSTED text/metadata: drop anything shaped like a prompt injection
        # BEFORE it can influence anything downstream. The url is attacker-
        # controlled just like the page body, so it is guarded too.
        if (
            _INJECTION.search(text)
            or _INJECTION.search(title)
            or _INJECTION.search(url)
        ):
            return None
        domain = _registrable_domain(url)
        if not domain:
            return None
        scrubbed = _scrub(text)
        clean_title = _scrub(title)
        if not scrubbed.strip():
            return None
        # The structural scrubber only masks long, well-formed tokens. Re-check
        # the scrubbed text/title with the shared lesson-safety gate so short
        # credential shapes (sk-{16..31}, gh_{20..35}), keyworded secrets and
        # heavy PII that slipped past the scrubber cause the whole result to be
        # dropped (fail-closed) rather than leaking into a claim.
        reject_text, _ = should_reject_lesson_value(scrubbed)
        if reject_text:
            return None
        if clean_title.strip():
            reject_title, _ = should_reject_lesson_value(clean_title)
            if reject_title:
                return None
        # Scrub provenance metadata too: a url query string or a published_at
        # value can carry a token.
        clean_url = _scrub(url)
        clean_published = (
            _scrub(published_at) if isinstance(published_at, str) else None
        )
        return {
            "url": clean_url,
            "title": clean_title,
            "published_at": clean_published,
            "domain": domain,
            "subject_key": _subject_key(scrubbed),
            "value_sig": _value_sig(scrubbed),
            "claim_text": scrubbed,
            "is_temporal": _is_temporal(text),
        }

    def _build_candidates(
        self, topic: str, groups: Dict[tuple, dict], fetched_at: str
    ) -> List[WebFactCandidate]:
        out: List[WebFactCandidate] = []
        expiry = _iso(_utc_now() + self._temporal_ttl)
        for grp in groups.values():
            source_count = len(grp["domains"])
            is_temporal = grp["is_temporal"]
            out.append(WebFactCandidate(
                topic=topic,
                subject_key=grp["subject_key"],
                value_sig=grp["value_sig"],
                claim_text=grp["claim_text"],
                state="candidate",  # invariant
                source_count=source_count,
                needs_more_sources=source_count < self.min_sources,
                contradicted=False,
                conflicting_values=[],
                is_temporal=is_temporal,
                expires_at=expiry if is_temporal else None,
                provenance=grp["provenance"],
                created_at=fetched_at,
            ))
        return out

    @staticmethod
    def _flag_contradictions(candidates: List[WebFactCandidate]) -> None:
        """Mark candidates that share a subject but assert different values.
        Contradicted candidates are held (fail-closed: not saveable)."""
        by_subject: Dict[str, set] = {}
        for c in candidates:
            by_subject.setdefault(c.subject_key, set()).add(c.value_sig)
        for c in candidates:
            values = by_subject.get(c.subject_key, set())
            if len(values) > 1:
                c.contradicted = True
                c.conflicting_values = sorted(v for v in values if v != c.value_sig)

    # -- save gate (fail-closed) ------------------------------------------
    def is_saveable(self, candidate: WebFactCandidate) -> bool:
        """Fail-closed gate for the Section-E persistence layer. A candidate is
        only eligible for the confirm path when it is corroborated by enough
        independent sources AND not contradicted. Even then it remains a
        ``candidate`` here — actual promotion to ``confirmed`` is Section-E's
        responsibility, not this module's."""
        return (
            candidate.state == "candidate"
            and not candidate.needs_more_sources
            and not candidate.contradicted
            and candidate.source_count >= self.min_sources
        )
