"""
Helper functions and data classes for eval tests.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
import os


# LLM-as-judge configuration
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-oss:20b")
JUDGE_BASE_URL = os.environ.get("EVAL_JUDGE_BASE_URL", "http://localhost:11434")


@dataclass
class MockConfig:
    """Minimal config object for eval tests."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    db_path: str = ":memory:"
    sqlite_vss_path: Optional[str] = None
    voice_debug: bool = True
    tts_enabled: bool = False
    tts_engine: str = "system"
    tts_voice: Optional[str] = None
    tts_rate: int = 200
    tts_chatterbox_device: str = "cpu"
    tts_chatterbox_audio_prompt: Optional[str] = None
    tts_chatterbox_exaggeration: float = 0.5
    tts_chatterbox_cfg_weight: float = 0.5
    web_search_enabled: bool = True
    llm_profile_select_timeout_sec: float = 10.0
    llm_tools_timeout_sec: float = 8.0
    llm_embed_timeout_sec: float = 10.0
    llm_chat_timeout_sec: float = 45.0
    agentic_max_turns: int = 8
    memory_enrichment_max_results: int = 5
    active_profiles: List[str] = field(default_factory=lambda: ["developer", "business", "life"])
    location_enabled: bool = True
    location_ip_address: Optional[str] = None
    location_auto_detect: bool = False
    location_cgnat_resolve_public_ip: bool = False
    dialogue_memory_timeout: int = 300
    mcps: Dict[str, Any] = field(default_factory=dict)
    use_stdin: bool = True


@dataclass
class EvalResult:
    """Result of a single eval test case."""
    query: str
    response: Optional[str]
    is_passed: bool
    failure_reason: Optional[str] = None
    tool_calls_made: List[str] = field(default_factory=list)
    turn_count: int = 0

    def __str__(self) -> str:
        status = "✅ PASS" if self.is_passed else "❌ FAIL"
        lines = [
            f"{status}: {self.query[:50]}...",
            f"  Response: {(self.response or '')[:100]}...",
            f"  Tools used: {', '.join(self.tool_calls_made) or 'none'}",
            f"  Turns: {self.turn_count}",
        ]
        if self.failure_reason:
            lines.append(f"  Reason: {self.failure_reason}")
        return "\n".join(lines)


@dataclass
class EvalCase:
    """A single eval test case definition."""
    name: str
    query: str
    expected_tool_calls: List[str] = field(default_factory=list)
    response_should_contain: List[str] = field(default_factory=list)
    response_should_not_contain: List[str] = field(default_factory=list)
    custom_validator: Optional[Callable[[str], bool]] = None
    profile_hint: Optional[str] = None


def assert_response_quality(result: EvalResult, case: EvalCase) -> None:
    """Assert that the response meets quality criteria."""
    response = result.response or ""
    response_lower = response.lower()

    # Check expected content
    for expected in case.response_should_contain:
        assert expected.lower() in response_lower, (
            f"Response should contain '{expected}' but got: {response[:200]}..."
        )

    # Check excluded content
    for excluded in case.response_should_not_contain:
        assert excluded.lower() not in response_lower, (
            f"Response should NOT contain '{excluded}' but got: {response[:200]}..."
        )

    # Check custom validator
    if case.custom_validator:
        assert case.custom_validator(response), (
            f"Custom validation failed for response: {response[:200]}..."
        )


def is_generic_greeting(response: str) -> bool:
    """Check if response is a generic greeting that ignores the query."""
    generic_patterns = [
        "how can i help you",
        "what can i do for you",
        "what would you like",
        "how may i assist",
        "is there something",
        "let me know what",
        "feel free to ask",
    ]
    response_lower = response.lower()
    return any(pattern in response_lower for pattern in generic_patterns)


def response_addresses_topic(response: str, topic_keywords: List[str]) -> bool:
    """Check if response addresses the topic by mentioning relevant keywords."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in topic_keywords)


def create_mock_llm_response(content: str, tool_calls: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Create a mock LLM response in Ollama format."""
    message = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message}


def create_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Create a tool call in OpenAI format."""
    return {
        "id": f"call_{name}_001",
        "function": {
            "name": name,
            "arguments": args
        }
    }


# =============================================================================
# LLM-as-Judge Evaluation
# =============================================================================

@dataclass
class JudgeVerdict:
    """Result from LLM judge evaluation."""
    is_passed: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)


def is_judge_llm_available() -> bool:
    """Check if the judge LLM is available and the model exists."""
    import requests
    try:
        # First check if Ollama is running
        resp = requests.get(f"{JUDGE_BASE_URL.rstrip('/')}/api/tags", timeout=2)
        if resp.status_code != 200:
            return False

        # Check if the judge model is available
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        # Check if our judge model (or a variant) is available
        judge_base = JUDGE_MODEL.split(":")[0]
        return any(judge_base in name for name in model_names)
    except Exception:
        return False


def call_judge_llm(system_prompt: str, user_prompt: str, timeout_sec: float = 30.0) -> Optional[str]:
    """Call the judge LLM with a prompt."""
    import requests

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"num_ctx": 4096},
    }

    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "")
    except Exception as e:
        print(f"⚠️ Judge LLM call failed: {e}")
        return None
    return None


def judge_response_answers_query(query: str, response: str, context: Optional[str] = None) -> JudgeVerdict:
    """
    Use LLM to judge if the response actually answers the user's query.

    Args:
        query: The user's original question
        response: The assistant's response
        context: Optional context about what data was available (e.g., tool results)

    Returns:
        JudgeVerdict with pass/fail, score, and reasoning
    """
    system_prompt = """You are an evaluation judge for a voice assistant. Your job is to determine if the assistant's response actually answers the user's question with real information.

Score the response on these criteria (0-10 each):
1. RELEVANCE: Does the response address the specific question asked? Score 0 if it doesn't mention the topic at all.
2. COMPLETENESS: Does it provide the information the user was seeking? Score 0 for empty acknowledgments like "Sure!", "OK!", "Got it!" that provide no actual information.
3. ACCURACY: Is the information factually plausible (based on any context provided)? Score 0 if no factual information is provided.
4. NO_DEFLECTION: Does it avoid generic greetings, deflections like "How can I help you?", or empty acknowledgments? Score 0 for responses under 20 characters that don't answer the question.

IMPORTANT: A response that just acknowledges without providing any actual information (e.g., "Sure thing!", "OK!", "Got it!") should score 0 on COMPLETENESS and fail overall.

Output your evaluation in this EXACT format:
RELEVANCE: [0-10]
COMPLETENESS: [0-10]
ACCURACY: [0-10]
NO_DEFLECTION: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    user_prompt = f"""User Query: {query}

Assistant Response: {response}"""

    if context:
        user_prompt += f"\n\nContext (data available to assistant):\n{context[:2000]}"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Fallback to heuristic evaluation if judge fails
        return JudgeVerdict(
            is_passed=not is_generic_greeting(response) and len(response) > 50,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    # Parse the judge response
    return _parse_judge_response(judge_response)


def judge_search_query_quality(
    user_query: str,
    search_query: str,
    location: Optional[str] = None,
    time_context: Optional[str] = None
) -> JudgeVerdict:
    """
    Use LLM to judge if the search query is well-formed for the user's intent.

    Args:
        user_query: What the user asked
        search_query: The search query the assistant generated
        location: User's known location (should be included if relevant)
        time_context: Time-related context (e.g., "this week", "tomorrow")

    Returns:
        JudgeVerdict evaluating search query quality
    """
    system_prompt = """You are evaluating search queries generated by a voice assistant.

Score the search query on these criteria (0-10 each):
1. INTENT_MATCH: Does the search query capture the user's actual intent?
2. LOCATION_AWARENESS: If location is known and relevant, is it included appropriately?
3. TIME_AWARENESS: If the query has time context, is it reflected in the search?
4. SPECIFICITY: Is the query specific enough to get useful results?

Output your evaluation in this EXACT format:
INTENT_MATCH: [0-10]
LOCATION_AWARENESS: [0-10]
TIME_AWARENESS: [0-10]
SPECIFICITY: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    user_prompt = f"""User Query: "{user_query}"
Generated Search Query: "{search_query}"
"""
    if location:
        user_prompt += f"User's Known Location: {location}\n"
    if time_context:
        user_prompt += f"Time Context: {time_context}\n"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Heuristic fallback
        has_location = location and any(
            loc_part.lower() in search_query.lower()
            for loc_part in location.split(",")[0].split()
        )
        return JudgeVerdict(
            is_passed=has_location if location else True,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    return _parse_judge_response(judge_response)


def _parse_judge_response(response: str) -> JudgeVerdict:
    """Parse the structured judge response into a JudgeVerdict."""
    lines = response.strip().split("\n")
    criteria_scores = {}
    is_passed = False
    reasoning = ""

    for line in lines:
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "OVERALL":
                is_passed = "PASS" in value.upper()
            elif key == "REASONING":
                reasoning = value
            else:
                # Try to parse as score
                try:
                    score = float(value.split()[0])
                    criteria_scores[key.lower()] = score / 10.0  # Normalize to 0-1
                except (ValueError, IndexError):
                    pass

    # Calculate average score
    avg_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0.5

    return JudgeVerdict(
        is_passed=is_passed,
        score=avg_score,
        reasoning=reasoning,
        criteria_scores=criteria_scores
    )


def judge_tool_usage_appropriateness(
    query: str,
    tools_called: List[str],
    tool_args: List[Dict[str, Any]],
    expected_tools: Optional[List[str]] = None
) -> JudgeVerdict:
    """
    Judge whether the tools used were appropriate for the query.

    Args:
        query: User's question
        tools_called: List of tool names that were called
        tool_args: List of arguments passed to each tool
        expected_tools: Optional list of tools that should have been called

    Returns:
        JudgeVerdict on tool usage
    """
    system_prompt = """You are evaluating tool usage by a voice assistant.

Score on these criteria (0-10 each):
1. TOOL_SELECTION: Were the right tools chosen for the task?
2. ARG_QUALITY: Were the tool arguments well-formed and appropriate?
3. EFFICIENCY: Was there unnecessary tool calling or missing necessary calls?

Output your evaluation in this EXACT format:
TOOL_SELECTION: [0-10]
ARG_QUALITY: [0-10]
EFFICIENCY: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    tool_info = "\n".join([
        f"- {name}: {args}" for name, args in zip(tools_called, tool_args)
    ]) if tools_called else "No tools called"

    user_prompt = f"""User Query: "{query}"

Tools Called:
{tool_info}
"""
    if expected_tools:
        user_prompt += f"\nExpected Tools: {', '.join(expected_tools)}"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Heuristic fallback
        has_expected = not expected_tools or all(t in tools_called for t in expected_tools)
        return JudgeVerdict(
            is_passed=has_expected,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    return _parse_judge_response(judge_response)

