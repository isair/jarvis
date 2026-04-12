"""
🧠 Knowledge Graph Operations — LLM-dependent graph logic.

Keeps graph.py as a pure data store (SQLite only). This module handles:
- Knowledge extraction from conversation summaries
- Best-node traversal (greedy descent via recent → top → root entry points)
- Auto-split when a node exceeds the token threshold

All LLM calls use call_llm_direct from the local Ollama instance.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..debug import debug_log
from ..llm import call_llm_direct
from .graph import GraphMemoryStore, MemoryNode, SPLIT_THRESHOLD, MAX_TRAVERSAL_DEPTH


# ── Memory extraction from dialogue ───────────────────────────────────


def extract_graph_memories(
    summary: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 30.0,
    thinking: bool = False,
    date_utc: Optional[str] = None,
) -> list[str]:
    """Extract novel knowledge from a conversation summary.

    The knowledge graph stores anything the assistant learned during
    conversations that it wouldn't already know — real-world facts,
    user-specific information, discovered details. This is the
    assistant's persistent world knowledge, not just facts about the user.

    Returns a list of self-contained factual statements,
    or an empty list if nothing novel was found.

    Args:
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Included as a date prefix on each fact for temporal context.
    """
    system_prompt = (
        "You extract NOVEL KNOWLEDGE learned during a conversation. "
        "This builds the assistant's persistent knowledge base — things "
        "it wouldn't already know. Each fact must be a self-contained "
        "statement that would be useful to recall in future conversations.\n\n"
        "EXTRACT — novel knowledge:\n"
        "- Facts about the user (preferences, habits, plans, life details, "
        "location, relationships)\n"
        "- Real-world discoveries (local businesses, opening hours, prices, "
        "specific venues)\n"
        "- Practical knowledge that was discovered (recipes that worked, "
        "techniques, solutions to specific problems)\n"
        "- Recent events or developments after your training cutoff\n"
        "- Corrections to what you'd normally assume\n\n"
        "DO NOT EXTRACT — these create noise or mislead:\n"
        "- What the ASSISTANT said, recommended, or suggested — only "
        "extract knowledge the USER brought or that was discovered, not "
        "the assistant's own advice back to itself\n"
        "- Weather or time-of-day snapshots — these go stale immediately "
        "and will confuse future lookups (e.g. do NOT store "
        '"today\'s weather is 20°C" or "the current date is Sunday")\n'
        "- Common knowledge you already have (how GitHub search works, "
        "basic nutrition guidelines, how Chinese names work, etc.)\n"
        "- Vague or content-free statements (\"user explored options\", "
        '"user engaged in an exchange")\n'
        "- Pure meta-interaction (greetings, thank-yous, user asked for "
        "a recap, user asked for the time)\n\n"
        "Write facts as KNOWLEDGE, not as interaction descriptions:\n"
        "Wrong: \"The assistant recommended adding vegetables\" — "
        "this is the assistant's own advice\n"
        "Wrong: \"User asked about boxing gyms\" — "
        "reframe as: \"Trenches Boxing Club in Hackney has evening classes\"\n"
        "Wrong: \"The current location is Tbilisi\" — "
        "reframe as: \"The user was in Tbilisi, Georgia in April 2026\"\n\n"
        "Respond with ONLY a JSON array of strings.\n"
        "If nothing novel was learned, respond with an empty array: []\n"
        'Example: ["The user follows an 1800 kcal daily meal plan", '
        '"Trenches Boxing Club in Hackney offers evening classes", '
        '"A soy-oyster-teriyaki glaze works well for air-fried chicken breast"]'
    )

    # Include date so each fact carries temporal context
    date_prefix = f"(Date: {date_utc}) " if date_utc else ""
    user_content = (
        f"Extract novel knowledge from this conversation summary:\n"
        f"{date_prefix}{summary}"
    )

    debug_log(f"graph memory extraction: sending {len(summary)} chars to {ollama_chat_model}", "memory")

    response = call_llm_direct(
        base_url=ollama_base_url,
        chat_model=ollama_chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )

    if not response:
        debug_log("graph memory extraction: LLM returned no response", "memory")
        return []

    debug_log(f"graph memory extraction: got response ({len(response)} chars)", "memory")

    # Parse JSON array from the response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        debug_log(f"graph memory extraction: no JSON array found in response: {response[:200]}", "memory")
        return []

    try:
        facts = json.loads(json_match.group())
        if not isinstance(facts, list):
            debug_log(f"graph memory extraction: parsed JSON is not a list: {type(facts)}", "memory")
            return []
        # Filter to non-empty strings
        facts = [str(f).strip() for f in facts if isinstance(f, str) and str(f).strip()]
        debug_log(f"graph memory extraction: got {len(facts)} facts", "memory")
        return facts
    except (json.JSONDecodeError, ValueError) as e:
        debug_log(f"graph memory extraction: JSON parse failed — {e}", "memory")
        return []


# ── Best-node traversal ───────────────────────────────────────────────


def _llm_pick_best_child(
    fragment: str,
    children: list[MemoryNode],
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 15.0,
    thinking: bool = False,
) -> Optional[str]:
    """Ask the LLM which child node best fits a memory fragment.

    Returns the chosen child's id, or None if none fit well.
    """
    if not children:
        return None

    options = []
    for i, child in enumerate(children, 1):
        options.append(f"{i}. {child.name}: {child.description}")
    options_text = "\n".join(options)

    system_prompt = (
        "You are a memory organiser. Given a fact to store and a list of "
        "category nodes, pick the single best-fitting category.\n"
        "If NONE of the categories fit well, respond with NONE.\n"
        "Respond with ONLY the number (1, 2, ...) or NONE. Nothing else."
    )
    user_content = (
        f"Fact to store: {fragment}\n\n"
        f"Categories:\n{options_text}"
    )

    response = call_llm_direct(
        base_url=ollama_base_url,
        chat_model=ollama_chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )

    if not response:
        return None

    response = response.strip().upper()
    if "NONE" in response:
        return None

    # Extract a number
    num_match = re.search(r'(\d+)', response)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(children):
            return children[idx].id

    return None


def find_best_node(
    store: GraphMemoryStore,
    fragment: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 15.0,
    thinking: bool = False,
) -> str:
    """Find the best node to store a memory fragment using three entry points.

    Traversal order:
    1. Recent nodes — check if fragment fits a recently accessed node
    2. Top nodes — check frequently accessed domains
    3. Root traversal — greedy top-down descent from root

    Returns the id of the best node.
    """
    # Entry point 1: Check recent nodes
    recent = store.get_recent_nodes(limit=5)
    if recent:
        best = _llm_pick_best_child(
            fragment, recent, ollama_base_url, ollama_chat_model,
            timeout_sec=timeout_sec, thinking=thinking,
        )
        if best is not None:
            debug_log(f"graph traversal: matched recent node {best[:8]}", "memory")
            return best

    # Entry point 2: Check top nodes (excluding any already checked as recent)
    recent_ids = {n.id for n in recent} if recent else set()
    top = [n for n in store.get_top_nodes(limit=10) if n.id not in recent_ids]
    if top:
        best = _llm_pick_best_child(
            fragment, top, ollama_base_url, ollama_chat_model,
            timeout_sec=timeout_sec, thinking=thinking,
        )
        if best is not None:
            debug_log(f"graph traversal: matched top node {best[:8]}", "memory")
            return best

    # Entry point 3: Greedy descent from root
    current_id = "root"
    for depth in range(MAX_TRAVERSAL_DEPTH):
        children = store.get_children(current_id)
        if not children:
            break  # Leaf node — write here

        best = _llm_pick_best_child(
            fragment, children, ollama_base_url, ollama_chat_model,
            timeout_sec=timeout_sec, thinking=thinking,
        )
        if best is None:
            break  # None of the children fit — write to current node
        current_id = best

    debug_log(f"graph traversal: writing to node {current_id[:8]} (depth {depth if 'depth' in dir() else 0})", "memory")
    return current_id


# ── Auto-split ─────────────────────────────────────────────────────────


def auto_split_node(
    store: GraphMemoryStore,
    node_id: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 45.0,
    thinking: bool = False,
) -> bool:
    """Split a node whose data exceeds SPLIT_THRESHOLD into child nodes.

    The LLM proposes 2-5 categories and distributes the facts among them.
    The parent node's data is cleared and its description updated to a summary.

    Returns True if the split succeeded.
    """
    node = store.get_node(node_id)
    if node is None or node.data_token_count <= SPLIT_THRESHOLD:
        return False

    debug_log(f"auto-split: node '{node.name}' ({node_id[:8]}) has {node.data_token_count} tokens", "memory")

    system_prompt = (
        "You are a knowledge organiser. A collection of facts has grown too "
        "large for a single node. Organise them into 2-5 categories.\n\n"
        "Rules:\n"
        "- Each fact must be assigned to exactly one category\n"
        "- Category names should be concise (2-4 words)\n"
        "- Descriptions should be 1-2 sentences explaining what the category covers\n\n"
        "Consolidation — apply while distributing:\n"
        "- Merge duplicate or near-duplicate facts into one\n"
        "- If repeated similar activities appear across different dates "
        "(e.g. ate X on Monday, ate X on Thursday), consolidate into a pattern "
        '(e.g. "Regularly eats X") — drop individual occurrences\n'
        "- Preserve date context only for significant events "
        "(e.g. started new job on 2025-03-01)\n\n"
        "Pruning — DROP facts that are common knowledge:\n"
        "- Remove anything you already know from your training data "
        "(e.g. general nutrition facts, well-known places, public figures' "
        "basic info, how-to steps for common tasks)\n"
        "- Only keep knowledge that is NOVEL to you: user-specific details, "
        "local/niche information, personal circumstances, recent events "
        "after your training cutoff, or corrections to what you'd assume\n"
        "- When in doubt, keep it — but actively look for things to prune\n\n"
        "Respond with ONLY valid JSON in this format:\n"
        '{"categories": [{"name": "Category Name", "description": "What this covers", '
        '"facts": ["fact 1", "fact 2"]}], "summary": "1-2 sentence summary of everything"}'
    )

    user_content = (
        f"Current node: {node.name}\n"
        f"Current description: {node.description}\n\n"
        f"Facts to organise:\n{node.data}"
    )

    response = call_llm_direct(
        base_url=ollama_base_url,
        chat_model=ollama_chat_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        thinking=thinking,
    )

    if not response:
        debug_log("auto-split: LLM returned no response", "memory")
        return False

    # Parse JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        debug_log("auto-split: no JSON found in response", "memory")
        return False

    try:
        result = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError) as e:
        debug_log(f"auto-split: JSON parse failed — {e}", "memory")
        return False

    categories = result.get("categories", [])
    summary = result.get("summary", node.description)

    # Validate: need at least 2 categories
    if len(categories) < 2:
        debug_log("auto-split: fewer than 2 categories proposed, aborting", "memory")
        return False

    # Validate: each category needs a name and at least one fact
    for cat in categories:
        if not cat.get("name") or not cat.get("facts"):
            debug_log(f"auto-split: invalid category {cat.get('name', '?')}, aborting", "memory")
            return False

    # Create child nodes
    for cat in categories:
        child_data = "\n".join(str(f) for f in cat["facts"])
        store.create_node(
            name=str(cat["name"]),
            description=str(cat.get("description", f"Memories about: {cat['name']}")),
            data=child_data,
            parent_id=node_id,
        )
        debug_log(f"  auto-split: created child '{cat['name']}' with {len(cat['facts'])} facts", "memory")

    # Clear parent data and update description to summary
    store.update_node(node_id, data="", description=str(summary))

    debug_log(f"auto-split: node '{node.name}' split into {len(categories)} children", "memory")
    return True


# ── Orchestrator ───────────────────────────────────────────────────────


def update_graph_from_dialogue(
    store: GraphMemoryStore,
    summary: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 30.0,
    thinking: bool = False,
    date_utc: Optional[str] = None,
) -> int:
    """End-to-end: extract memories from a summary, place each in the best
    node, and trigger auto-split if needed.

    Args:
        date_utc: Optional date string (YYYY-MM-DD) for the diary entry.
            Passed to extraction to help distinguish daily events from enduring facts.

    Returns the number of facts stored.
    """
    # Step 1: Extract discrete facts from the conversation summary
    facts = extract_graph_memories(
        summary=summary,
        ollama_base_url=ollama_base_url,
        ollama_chat_model=ollama_chat_model,
        timeout_sec=timeout_sec,
        thinking=thinking,
        date_utc=date_utc,
    )

    if not facts:
        debug_log("graph update: no facts extracted from summary", "memory")
        return 0

    stored = 0
    for fact in facts:
        try:
            # Step 2: Find the best node for this fact
            node_id = find_best_node(
                store=store,
                fragment=fact,
                ollama_base_url=ollama_base_url,
                ollama_chat_model=ollama_chat_model,
                timeout_sec=15.0,
                thinking=thinking,
            )

            # Step 3: Append the fact to the chosen node
            threshold_exceeded = store.append_to_node(node_id, fact)
            store.touch_node(node_id)
            stored += 1

            # Step 4: Auto-split if the node has grown too large
            if threshold_exceeded:
                auto_split_node(
                    store=store,
                    node_id=node_id,
                    ollama_base_url=ollama_base_url,
                    ollama_chat_model=ollama_chat_model,
                    timeout_sec=45.0,
                    thinking=thinking,
                )

        except Exception as e:
            debug_log(f"graph update: failed to store fact — {e}", "memory")
            continue

    debug_log(f"graph update: stored {stored}/{len(facts)} facts", "memory")
    return stored
