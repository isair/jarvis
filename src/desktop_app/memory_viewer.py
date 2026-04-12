"""
🧠 Jarvis Memory Viewer

A beautiful web interface for exploring Jarvis's conversation memories.
Run directly: python -m desktop_app.memory_viewer
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, Response

from jarvis.config import load_settings
from jarvis.debug import debug_log
from jarvis.memory.graph import GraphMemoryStore


app = Flask(__name__)

# Global database connection
_db_conn: Optional[sqlite3.Connection] = None
_graph_store: Optional[GraphMemoryStore] = None


def _get_db_path() -> str:
    """Get the database path from settings."""
    try:
        settings = load_settings()
        return settings.db_path
    except Exception:
        # Fallback to default path
        base = Path.home() / ".local" / "share" / "jarvis"
        return str(base / "jarvis.db")


def get_db() -> sqlite3.Connection:
    """Get or create database connection."""
    global _db_conn
    if _db_conn is None:
        db_path = _get_db_path()
        _db_conn = sqlite3.connect(db_path, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
    return _db_conn


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert sqlite3.Row to dictionary."""
    return {key: row[key] for key in row.keys()}


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/memories")
def get_memories() -> Response:
    """
    Get all conversation summaries with optional filtering.

    Query params:
    - search: Search query for full-text search
    - topic: Filter by topic (comma-separated for multiple)
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - limit: Max results (default 100)
    """
    conn = get_db()
    cur = conn.cursor()

    search = request.args.get("search", "").strip()
    topic_filter = request.args.get("topic", "").strip()
    from_date = request.args.get("from_date", "").strip()
    to_date = request.args.get("to_date", "").strip()
    limit = min(int(request.args.get("limit", 100)), 500)

    params: list[Any] = []
    conditions: list[str] = []

    # Build query based on filters
    if search:
        # Use FTS for search
        conditions.append("cs.id IN (SELECT rowid FROM summaries_fts WHERE summaries_fts MATCH ?)")
        params.append(search)

    if topic_filter:
        # Filter by topic(s)
        topics = [t.strip().lower() for t in topic_filter.split(",") if t.strip()]
        if topics:
            topic_conditions = " OR ".join(["LOWER(cs.topics) LIKE ?" for _ in topics])
            conditions.append(f"({topic_conditions})")
            params.extend([f"%{t}%" for t in topics])

    if from_date:
        conditions.append("cs.date_utc >= ?")
        params.append(from_date)

    if to_date:
        conditions.append("cs.date_utc <= ?")
        params.append(to_date)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT cs.id, cs.date_utc, cs.ts_utc, cs.summary, cs.topics, cs.source_app
        FROM conversation_summaries cs
        WHERE {where_clause}
        ORDER BY cs.date_utc DESC
        LIMIT ?
    """
    params.append(limit)

    try:
        rows = cur.execute(query, params).fetchall()
        memories = [row_to_dict(row) for row in rows]

        # Parse topics into arrays
        for memory in memories:
            if memory.get("topics"):
                memory["topics_list"] = [t.strip() for t in memory["topics"].split(",") if t.strip()]
            else:
                memory["topics_list"] = []

        return jsonify({"memories": memories, "count": len(memories)})
    except Exception as e:
        return jsonify({"error": str(e), "memories": [], "count": 0}), 500


@app.route("/api/topics")
def get_topics() -> Response:
    """Get all unique topics with their counts."""
    conn = get_db()
    cur = conn.cursor()

    try:
        rows = cur.execute("""
            SELECT topics FROM conversation_summaries WHERE topics IS NOT NULL AND topics != ''
        """).fetchall()

        topic_counts: dict[str, int] = {}
        for row in rows:
            topics_str = row["topics"]
            for topic in topics_str.split(","):
                topic = topic.strip().lower()
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Sort by count descending
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "topics": [{"name": name, "count": count} for name, count in sorted_topics]
        })
    except Exception as e:
        return jsonify({"error": str(e), "topics": []}), 500


@app.route("/api/meals")
def get_meals() -> Response:
    """
    Get meal logs with optional date filtering.

    Query params:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - limit: Max results (default 100)
    """
    conn = get_db()
    cur = conn.cursor()

    from_date = request.args.get("from_date", "").strip()
    to_date = request.args.get("to_date", "").strip()
    limit = min(int(request.args.get("limit", 100)), 500)

    params: list[Any] = []
    conditions: list[str] = []

    if from_date:
        conditions.append("date(ts_utc) >= ?")
        params.append(from_date)

    if to_date:
        conditions.append("date(ts_utc) <= ?")
        params.append(to_date)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT * FROM meals
        WHERE {where_clause}
        ORDER BY ts_utc DESC
        LIMIT ?
    """
    params.append(limit)

    try:
        rows = cur.execute(query, params).fetchall()
        meals = [row_to_dict(row) for row in rows]
        return jsonify({"meals": meals, "count": len(meals)})
    except Exception as e:
        return jsonify({"error": str(e), "meals": [], "count": 0}), 500


@app.route("/api/stats")
def get_stats() -> Response:
    """Get memory statistics."""
    conn = get_db()
    cur = conn.cursor()

    try:
        # Total memories
        total_memories = cur.execute("SELECT COUNT(*) as count FROM conversation_summaries").fetchone()["count"]

        # Date range
        date_range = cur.execute("""
            SELECT MIN(date_utc) as earliest, MAX(date_utc) as latest
            FROM conversation_summaries
        """).fetchone()

        # Memories by month
        monthly_stats = cur.execute("""
            SELECT strftime('%Y-%m', date_utc) as month, COUNT(*) as count
            FROM conversation_summaries
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """).fetchall()

        # Total meals
        total_meals = cur.execute("SELECT COUNT(*) as count FROM meals").fetchone()["count"]

        return jsonify({
            "total_memories": total_memories,
            "earliest_date": date_range["earliest"],
            "latest_date": date_range["latest"],
            "monthly_stats": [row_to_dict(row) for row in monthly_stats],
            "total_meals": total_meals
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/<int:memory_id>")
def get_memory(memory_id: int) -> Response:
    """Get a single memory by ID."""
    conn = get_db()
    cur = conn.cursor()

    try:
        row = cur.execute("""
            SELECT * FROM conversation_summaries WHERE id = ?
        """, (memory_id,)).fetchone()

        if row:
            memory = row_to_dict(row)
            if memory.get("topics"):
                memory["topics_list"] = [t.strip() for t in memory["topics"].split(",") if t.strip()]
            else:
                memory["topics_list"] = []
            return jsonify({"memory": memory})
        else:
            return jsonify({"error": "Memory not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memory/<int:memory_id>", methods=["DELETE"])
def delete_memory(memory_id: int) -> Response:
    """Delete a memory by ID."""
    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM conversation_summaries WHERE id = ?", (memory_id,))
        conn.commit()

        if cur.rowcount > 0:
            return jsonify({"success": True, "message": "Memory deleted"})
        else:
            return jsonify({"error": "Memory not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/meal/<int:meal_id>", methods=["DELETE"])
def delete_meal(meal_id: int) -> Response:
    """Delete a meal by ID."""
    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
        conn.commit()

        if cur.rowcount > 0:
            return jsonify({"success": True, "message": "Meal deleted"})
        else:
            return jsonify({"error": "Meal not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Graph Memory (v2) API
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_store() -> GraphMemoryStore:
    """Get or create the graph memory store (shares the same DB)."""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphMemoryStore(_get_db_path())
    return _graph_store


@app.route("/api/graph/nodes")
def graph_get_all_nodes() -> Response:
    """Get all nodes for the graph visualisation."""
    store = get_graph_store()
    try:
        root_id = request.args.get("root", "root")
        max_depth = min(int(request.args.get("max_depth", 10)), 20)
        data = store.get_graph_data(root_id, max_depth=max_depth)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/tree")
def graph_get_tree() -> Response:
    """Get the full tree structure for the sidebar."""
    store = get_graph_store()
    try:
        root_id = request.args.get("root", "root")
        max_depth = min(int(request.args.get("max_depth", 10)), 20)
        tree = store.get_subtree(root_id, max_depth=max_depth)
        return jsonify(tree)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/node/<node_id>")
def graph_get_node(node_id: str) -> Response:
    """Get a single node with its children and ancestors."""
    store = get_graph_store()
    try:
        node = store.get_node(node_id)
        if node is None:
            return jsonify({"error": "Node not found"}), 404

        store.touch_node(node_id)
        children = store.get_children(node_id)
        ancestors = store.get_ancestors(node_id)

        return jsonify({
            "node": node.to_dict(),
            "children": [c.to_dict() for c in children],
            "ancestors": [a.to_dict() for a in ancestors],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/node", methods=["POST"])
def graph_create_node() -> Response:
    """Create a new memory node."""
    store = get_graph_store()
    try:
        body = request.get_json()
        if not body or not body.get("name"):
            return jsonify({"error": "name is required"}), 400

        # Validate field types
        name = body["name"]
        description = body.get("description", "")
        data = body.get("data", "")
        parent_id = body.get("parent_id", "root")
        if not isinstance(name, str) or not isinstance(description, str) \
                or not isinstance(data, str) or not isinstance(parent_id, str):
            return jsonify({"error": "name, description, data, and parent_id must be strings"}), 400

        node = store.create_node(
            name=name,
            description=description,
            data=data,
            parent_id=parent_id,
        )
        return jsonify({"node": node.to_dict()}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/node/<node_id>", methods=["PUT"])
def graph_update_node(node_id: str) -> Response:
    """Update an existing memory node."""
    store = get_graph_store()
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "Request body is required"}), 400

        kwargs = {}
        for field in ("name", "description", "data", "parent_id"):
            if field in body:
                if not isinstance(body[field], str):
                    return jsonify({"error": f"{field} must be a string"}), 400
                kwargs[field] = body[field]

        node = store.update_node(node_id, **kwargs)
        if node is None:
            return jsonify({"error": "Node not found or invalid parent"}), 404

        return jsonify({"node": node.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/node/<node_id>", methods=["DELETE"])
def graph_delete_node(node_id: str) -> Response:
    """Delete a memory node."""
    store = get_graph_store()
    try:
        if node_id == "root":
            return jsonify({"error": "Cannot delete root node"}), 400

        deleted = store.delete_node(node_id)
        if deleted:
            return jsonify({"success": True})
        return jsonify({"error": "Node not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/recent")
def graph_recent_nodes() -> Response:
    """Get recently accessed nodes."""
    store = get_graph_store()
    try:
        limit = min(int(request.args.get("limit", 10)), 50)
        nodes = store.get_recent_nodes(limit)
        return jsonify({"nodes": [n.to_dict() for n in nodes]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/top")
def graph_top_nodes() -> Response:
    """Get most frequently accessed nodes."""
    store = get_graph_store()
    try:
        limit = min(int(request.args.get("limit", 15)), 50)
        nodes = store.get_top_nodes(limit)
        return jsonify({"nodes": [n.to_dict() for n in nodes]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/stats")
def graph_stats() -> Response:
    """Get graph memory statistics."""
    store = get_graph_store()
    try:
        return jsonify({
            "total_nodes": store.get_node_count(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/import-diary", methods=["POST"])
def graph_import_diary() -> Response:
    """Import all diary conversation summaries into the graph memory system.

    Processes each summary through the extract → traverse → append → split
    pipeline. Returns a streaming response with progress updates so the UI
    can show real-time feedback.
    """
    from jarvis.config import load_settings
    from jarvis.memory.db import Database
    from jarvis.memory.graph_ops import update_graph_from_dialogue

    def generate():
        try:
            settings = load_settings()
            db_path = _get_db_path()
            db = Database(db_path, sqlite_vss_path=None)

            summaries = db.get_all_conversation_summaries()
            total = len(summaries)

            if total == 0:
                yield json.dumps({"type": "complete", "message": "No diary entries found to import.", "processed": 0, "total": 0}) + "\n"
                return

            yield json.dumps({"type": "start", "total": total}) + "\n"

            store = get_graph_store()
            processed = 0
            total_facts = 0

            for row in summaries:
                summary_text = row["summary"]
                date_utc = row["date_utc"]
                error_msg = None

                try:
                    debug_log(f"graph import: processing {date_utc} ({len(summary_text)} chars)", "memory")
                    facts_stored = update_graph_from_dialogue(
                        store=store,
                        summary=summary_text,
                        ollama_base_url=settings.ollama_base_url,
                        ollama_chat_model=settings.ollama_chat_model,
                        timeout_sec=settings.llm_chat_timeout_sec,
                        thinking=getattr(settings, 'llm_thinking_enabled', False),
                        date_utc=date_utc,
                    )
                    total_facts += facts_stored
                except Exception as e:
                    debug_log(f"graph import: failed for {date_utc} — {e}", "memory")
                    facts_stored = 0
                    error_msg = str(e)

                processed += 1
                progress_msg = {
                    "type": "progress",
                    "processed": processed,
                    "total": total,
                    "date": date_utc,
                    "facts": facts_stored,
                }
                if error_msg:
                    progress_msg["error"] = error_msg
                yield json.dumps(progress_msg) + "\n"

            yield json.dumps({
                "type": "complete",
                "message": f"Imported {total_facts} facts from {total} diary entries.",
                "processed": processed,
                "total": total,
                "total_facts": total_facts,
            }) + "\n"

            db.close()

        except Exception as e:
            debug_log(f"graph import failed: {e}", "memory")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> str:
    """Serve the memory viewer frontend."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Jarvis Memory</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Deep space theme with amber accents */
            --bg-primary: #0a0b0f;
            --bg-secondary: #12141a;
            --bg-tertiary: #1a1d26;
            --bg-card: #161920;
            --bg-hover: #1e222c;

            --accent-primary: #f59e0b;
            --accent-secondary: #fbbf24;
            --accent-glow: rgba(245, 158, 11, 0.15);
            --accent-muted: #92400e;

            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;

            --border-color: #27272a;
            --border-glow: rgba(245, 158, 11, 0.3);

            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;

            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
            --radius-xl: 24px;

            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 40px var(--accent-glow);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', '.AppleSystemUIFont', 'Segoe UI', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background:
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(245, 158, 11, 0.08), transparent),
                radial-gradient(ellipse 60% 40% at 100% 100%, rgba(139, 92, 246, 0.05), transparent);
            pointer-events: none;
            z-index: -1;
        }

        /* Header */
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 11, 15, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
        }

        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 2rem;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            font-size: 1.75rem;
            animation: pulse-glow 3s ease-in-out infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { filter: drop-shadow(0 0 8px var(--accent-glow)); }
            50% { filter: drop-shadow(0 0 16px var(--accent-primary)); }
        }

        .logo h1 {
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Search bar */
        .search-container {
            flex: 1;
            max-width: 500px;
        }

        .search-wrapper {
            position: relative;
        }

        .search-input {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 3rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        .search-input:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .search-icon {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        /* Stats badges */
        .stats-badges {
            display: flex;
            gap: 1rem;
        }

        .stat-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .stat-badge .value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--accent-secondary);
        }

        /* Main layout */
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 2rem;
        }

        /* Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .sidebar-section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
        }

        .sidebar-title {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Date filters */
        .date-filters {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .date-input {
            padding: 0.6rem 0.85rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        .date-input:focus {
            outline: none;
            border-color: var(--accent-primary);
        }

        /* Topic tags */
        .topics-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            max-height: 300px;
            overflow-y: auto;
        }

        .topic-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-xl);
            font-size: 0.8rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .topic-tag:hover {
            background: var(--bg-hover);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .topic-tag.active {
            background: var(--accent-glow);
            border-color: var(--accent-primary);
            color: var(--accent-secondary);
        }

        .topic-count {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Memory list */
        .memory-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .memory-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .memory-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, var(--accent-primary), var(--accent-muted));
            opacity: 0;
            transition: opacity 0.2s ease;
        }

        .memory-card:hover {
            border-color: var(--border-glow);
            transform: translateX(4px);
            box-shadow: var(--shadow-md);
        }

        .memory-card:hover::before {
            opacity: 1;
        }

        .memory-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }

        .memory-date {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-secondary);
        }

        .memory-actions {
            display: flex;
            gap: 0.5rem;
            opacity: 0;
            transition: opacity 0.15s ease;
        }

        .memory-card:hover .memory-actions {
            opacity: 1;
        }

        .action-btn {
            padding: 0.4rem 0.6rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.15s ease;
        }

        .action-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .action-btn.delete:hover {
            background: rgba(239, 68, 68, 0.15);
            border-color: var(--error);
            color: var(--error);
        }

        .memory-summary {
            font-size: 0.95rem;
            line-height: 1.7;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }

        .memory-topics {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }

        .memory-topic {
            padding: 0.25rem 0.6rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }

        .empty-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .empty-title {
            font-size: 1.25rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--bg-tertiary);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .tab {
            padding: 0.6rem 1.25rem;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tab:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .tab.active {
            background: var(--accent-glow);
            border-color: var(--accent-primary);
            color: var(--accent-secondary);
        }

        /* Meal cards */
        .meal-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 1rem;
            align-items: center;
            transition: all 0.2s ease;
        }

        .meal-card:hover {
            border-color: var(--border-glow);
        }

        .meal-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .meal-info h3 {
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .meal-delete {
            opacity: 0;
            transition: opacity 0.15s ease;
            flex-shrink: 0;
        }

        .meal-card:hover .meal-delete {
            opacity: 1;
        }

        .meal-time {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .meal-macros {
            display: flex;
            gap: 0.75rem;
        }

        .macro {
            text-align: center;
            padding: 0.5rem 0.75rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
        }

        .macro-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            font-size: 1rem;
            color: var(--accent-secondary);
        }

        .macro-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        /* Responsive */
        @media (max-width: 900px) {
            .main-container {
                grid-template-columns: 1fr;
            }

            .sidebar {
                order: 1;
            }

            .stats-badges {
                display: none;
            }
        }

        @media (max-width: 600px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }

            .search-container {
                max-width: none;
                width: 100%;
            }

            .main-container {
                padding: 1rem;
            }
        }

        /* Toast notifications */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-lg);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slide-in 0.3s ease;
            z-index: 1000;
        }

        .toast.success {
            border-color: var(--success);
        }

        .toast.error {
            border-color: var(--error);
        }

        @keyframes slide-in {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
        }

        /* ─── Graph Explorer ─── */
        .graph-explorer {
            display: grid;
            grid-template-columns: 240px 1fr 320px;
            gap: 0;
            height: calc(100vh - 200px);
            min-height: 500px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            overflow: hidden;
            background: var(--bg-card);
        }

        .graph-tree-sidebar {
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .tree-header {
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-shrink: 0;
        }

        .tree-container {
            flex: 1;
            overflow-y: auto;
            padding: 0.5rem 0;
        }

        .tree-node {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            cursor: pointer;
            font-size: 0.85rem;
            color: var(--text-secondary);
            transition: all 0.1s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .tree-node:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .tree-node.selected {
            background: var(--accent-glow);
            color: var(--accent-secondary);
        }

        .tree-toggle {
            width: 16px;
            height: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.6rem;
            color: var(--text-muted);
            flex-shrink: 0;
            transition: transform 0.15s ease;
        }

        .tree-toggle.expanded {
            transform: rotate(90deg);
        }

        .tree-toggle.leaf {
            visibility: hidden;
        }

        .tree-children {
            padding-left: 1rem;
        }

        .tree-children.collapsed {
            display: none;
        }

        .tree-node-name {
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .tree-node-count {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
            margin-left: auto;
            flex-shrink: 0;
            padding: 0 0.35rem;
            background: var(--bg-tertiary);
            border-radius: var(--radius-sm);
        }

        /* Graph canvas */
        .graph-canvas-container {
            position: relative;
            overflow: hidden;
            background: var(--bg-primary);
            background-image:
                radial-gradient(circle, var(--border-color) 1px, transparent 1px);
            background-size: 30px 30px;
        }

        #graph-canvas {
            width: 100%;
            height: 100%;
            display: block;
            cursor: grab;
        }

        #graph-canvas:active {
            cursor: grabbing;
        }

        .graph-toolbar {
            position: absolute;
            top: 0.75rem;
            left: 0.75rem;
            display: flex;
            gap: 0.35rem;
            z-index: 10;
        }

        .graph-btn {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.1s ease;
        }

        .graph-btn:hover {
            background: var(--bg-hover);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        /* Detail sidebar */
        .graph-detail-sidebar {
            border-left: 1px solid var(--border-color);
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .detail-empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            text-align: center;
            gap: 0.5rem;
        }

        .detail-empty .empty-icon {
            font-size: 2.5rem;
            opacity: 0.4;
        }

        .detail-breadcrumb {
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .detail-breadcrumb span {
            cursor: pointer;
            transition: color 0.1s;
        }

        .detail-breadcrumb span:hover {
            color: var(--accent-secondary);
        }

        .detail-breadcrumb .sep {
            cursor: default;
        }

        .detail-breadcrumb .sep:hover {
            color: var(--text-muted);
        }

        .detail-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .detail-description {
            font-size: 0.9rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .detail-section {
            margin-top: 0.5rem;
        }

        .detail-section-title {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .detail-data {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            line-height: 1.7;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 0.75rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }

        .detail-meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }

        .detail-meta-item {
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            padding: 0.5rem 0.65rem;
        }

        .detail-meta-label {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .detail-meta-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--accent-secondary);
        }

        .detail-children-list {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .detail-child {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.65rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all 0.1s ease;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .detail-child:hover {
            background: var(--bg-hover);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .detail-child-name {
            font-weight: 500;
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .detail-actions {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .detail-action-btn {
            flex: 1;
            min-width: 80px;
            padding: 0.5rem 0.75rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.8rem;
            text-align: center;
            transition: all 0.1s ease;
        }

        .detail-action-btn:hover {
            background: var(--bg-hover);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .detail-action-btn.delete:hover {
            background: rgba(239, 68, 68, 0.15);
            border-color: var(--error);
            color: var(--error);
        }

        /* Edit form in detail sidebar */
        .detail-edit-field {
            width: 100%;
            padding: 0.5rem 0.65rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            resize: vertical;
        }

        .detail-edit-field:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        textarea.detail-edit-field {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            min-height: 80px;
        }

        /* Node creation modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            width: 400px;
            max-width: 90vw;
            box-shadow: var(--shadow-lg);
        }

        .modal h3 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .modal-field {
            margin-bottom: 0.75rem;
        }

        .modal-field label {
            display: block;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .modal-actions {
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
            margin-top: 1rem;
        }

        .modal-btn {
            padding: 0.5rem 1.25rem;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.1s ease;
        }

        .modal-btn.primary {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
            color: var(--bg-primary);
            font-weight: 600;
        }

        .modal-btn.primary:hover {
            background: var(--accent-secondary);
        }

        .modal-btn.secondary {
            background: var(--bg-secondary);
            color: var(--text-secondary);
        }

        .modal-btn.secondary:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        /* Responsive graph */
        @media (max-width: 1100px) {
            .graph-explorer {
                grid-template-columns: 200px 1fr 260px;
            }
        }

        @media (max-width: 800px) {
            .graph-explorer {
                grid-template-columns: 1fr;
                grid-template-rows: 200px 1fr;
                height: calc(100vh - 180px);
            }
            .graph-tree-sidebar {
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }
            .graph-detail-sidebar {
                display: none;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <span class="logo-icon">🧠</span>
                <h1>Jarvis Memory</h1>
            </div>

            <div class="search-container">
                <div class="search-wrapper">
                    <span class="search-icon">🔍</span>
                    <input type="text" class="search-input" id="search-input" placeholder="Search memories..." />
                </div>
            </div>

            <div class="stats-badges">
                <div class="stat-badge">
                    <span>📝</span>
                    <span class="value" id="stats-memories">-</span>
                    <span>diary</span>
                </div>
                <div class="stat-badge">
                    <span>🧠</span>
                    <span class="value" id="stats-nodes">-</span>
                    <span>nodes</span>
                </div>
                <div class="stat-badge">
                    <span>🍽️</span>
                    <span class="value" id="stats-meals">-</span>
                    <span>meals</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-container">
        <aside class="sidebar">
            <div class="sidebar-section" id="date-filter-section">
                <div class="sidebar-title">📅 Date Range</div>
                <div class="date-filters">
                    <input type="date" class="date-input" id="from-date" placeholder="From" />
                    <input type="date" class="date-input" id="to-date" placeholder="To" />
                </div>
            </div>

            <div class="sidebar-section" id="topics-filter-section">
                <div class="sidebar-title">🏷️ Topics</div>
                <div class="topics-cloud" id="topics-cloud">
                    <div class="loading"><div class="spinner"></div></div>
                </div>
            </div>
        </aside>

        <section class="content">
            <div class="tabs">
                <button class="tab active" data-tab="memories">
                    <span>💭</span> Diary
                </button>
                <button class="tab" data-tab="graph">
                    <span>🧠</span> Knowledge
                </button>
                <button class="tab" data-tab="meals">
                    <span>🍽️</span> Meals
                </button>
            </div>

            <div id="memories-content" class="memory-list">
                <div class="loading"><div class="spinner"></div></div>
            </div>

            <div id="graph-content" style="display: none;">
                <div class="graph-explorer">
                    <!-- Left sidebar: tree navigator -->
                    <div class="graph-tree-sidebar">
                        <div class="tree-header">
                            <span>🌳</span> Memory Tree
                        </div>
                        <div class="tree-container" id="tree-container">
                            <div class="loading"><div class="spinner"></div></div>
                        </div>
                    </div>

                    <!-- Centre: graph canvas -->
                    <div class="graph-canvas-container">
                        <div class="graph-toolbar">
                            <button class="graph-btn" id="btn-zoom-in" title="Zoom in">➕</button>
                            <button class="graph-btn" id="btn-zoom-out" title="Zoom out">➖</button>
                            <button class="graph-btn" id="btn-fit" title="Fit to view">📐</button>
                            <button class="graph-btn" id="btn-add-node" title="Add node">✨</button>
                            <button class="graph-btn" id="btn-import-diary" title="Import from Diary">📥</button>
                        </div>
                        <canvas id="graph-canvas"></canvas>
                    </div>

                    <!-- Right sidebar: node details -->
                    <div class="graph-detail-sidebar" id="detail-sidebar">
                        <div class="detail-empty">
                            <div class="empty-icon">🧠</div>
                            <p>Select a node to view its details</p>
                        </div>
                    </div>
                </div>
            </div>

            <div id="meals-content" class="memory-list" style="display: none;">
                <div class="loading"><div class="spinner"></div></div>
            </div>
        </section>
    </main>

    <script>
        // State
        let currentTab = 'memories';
        let selectedTopics = new Set();
        let searchQuery = '';
        let fromDate = '';
        let toDate = '';
        let searchDebounce = null;

        // DOM Elements
        const searchInput = document.getElementById('search-input');
        const fromDateInput = document.getElementById('from-date');
        const toDateInput = document.getElementById('to-date');
        const topicsCloud = document.getElementById('topics-cloud');
        const memoriesContent = document.getElementById('memories-content');
        const mealsContent = document.getElementById('meals-content');
        const graphContent = document.getElementById('graph-content');
        const tabs = document.querySelectorAll('.tab');

        // Shared utilities
        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                      .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
        }

        // API calls
        async function fetchMemories() {
            const params = new URLSearchParams();
            if (searchQuery) params.set('search', searchQuery);
            if (selectedTopics.size > 0) params.set('topic', Array.from(selectedTopics).join(','));
            if (fromDate) params.set('from_date', fromDate);
            if (toDate) params.set('to_date', toDate);

            const response = await fetch('/api/memories?' + params);
            return response.json();
        }

        async function fetchTopics() {
            const response = await fetch('/api/topics');
            return response.json();
        }

        async function fetchMeals() {
            const params = new URLSearchParams();
            if (fromDate) params.set('from_date', fromDate);
            if (toDate) params.set('to_date', toDate);

            const response = await fetch('/api/meals?' + params);
            return response.json();
        }

        async function fetchStats() {
            const response = await fetch('/api/stats');
            return response.json();
        }

        async function deleteMemory(id) {
            const response = await fetch('/api/memory/' + id, { method: 'DELETE' });
            return response.json();
        }

        async function deleteMeal(id) {
            const response = await fetch('/api/meal/' + id, { method: 'DELETE' });
            return response.json();
        }

        // Render functions
        function renderTopics(topics) {
            if (!topics.length) {
                topicsCloud.innerHTML = '<div class="empty-state"><p>No topics yet</p></div>';
                return;
            }

            topicsCloud.innerHTML = topics.map(topic => `
                <button class="topic-tag ${selectedTopics.has(topic.name) ? 'active' : ''}"
                        data-topic="${escapeHtml(topic.name)}">
                    ${escapeHtml(topic.name)}
                    <span class="topic-count">${topic.count}</span>
                </button>
            `).join('');

            // Add click handlers
            topicsCloud.querySelectorAll('.topic-tag').forEach(tag => {
                tag.addEventListener('click', () => {
                    const topic = tag.dataset.topic;
                    if (selectedTopics.has(topic)) {
                        selectedTopics.delete(topic);
                    } else {
                        selectedTopics.add(topic);
                    }
                    renderTopics(topics);
                    loadMemories();
                });
            });
        }

        function formatDate(dateStr) {
            const date = new Date(dateStr + 'T00:00:00');
            const now = new Date();
            const diff = Math.floor((now - date) / (1000 * 60 * 60 * 24));

            if (diff === 0) return 'Today';
            if (diff === 1) return 'Yesterday';
            if (diff < 7) return `${diff} days ago`;

            return date.toLocaleDateString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
            });
        }

        function renderMemories(memories) {
            if (!memories.length) {
                memoriesContent.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">🌙</div>
                        <div class="empty-title">No memories found</div>
                        <p>Try adjusting your search or filters</p>
                    </div>
                `;
                return;
            }

            memoriesContent.innerHTML = memories.map(memory => `
                <article class="memory-card" data-id="${memory.id}">
                    <div class="memory-header">
                        <div class="memory-date">
                            <span>📅</span>
                            ${formatDate(memory.date_utc)}
                        </div>
                        <div class="memory-actions">
                            <button class="action-btn delete" title="Delete memory">🗑️</button>
                        </div>
                    </div>
                    <p class="memory-summary">${escapeHtml(memory.summary)}</p>
                    ${memory.topics_list.length ? `
                        <div class="memory-topics">
                            ${memory.topics_list.map(t => `<span class="memory-topic">${escapeHtml(t)}</span>`).join('')}
                        </div>
                    ` : ''}
                </article>
            `).join('');

            // Add delete handlers
            memoriesContent.querySelectorAll('.action-btn.delete').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const card = e.target.closest('.memory-card');
                    const id = card.dataset.id;

                    if (confirm('Delete this memory?')) {
                        const result = await deleteMemory(id);
                        if (result.success) {
                            card.remove();
                            showToast('Memory deleted', 'success');
                            loadStats();
                        } else {
                            showToast('Failed to delete', 'error');
                        }
                    }
                });
            });
        }

        function renderMeals(meals) {
            if (!meals.length) {
                mealsContent.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">🍽️</div>
                        <div class="empty-title">No meals logged</div>
                        <p>Meal tracking data will appear here</p>
                    </div>
                `;
                return;
            }

            mealsContent.innerHTML = meals.map(meal => `
                <div class="meal-card" data-id="${meal.id}">
                    <div class="meal-info">
                        <div class="meal-header">
                            <h3>${meal.description}</h3>
                            <button class="action-btn delete meal-delete" title="Delete meal">🗑️</button>
                        </div>
                        <div class="meal-time">${new Date(meal.ts_utc).toLocaleString()}</div>
                    </div>
                    <div class="meal-macros">
                        ${meal.calories_kcal ? `
                            <div class="macro">
                                <div class="macro-value">${Math.round(meal.calories_kcal)}</div>
                                <div class="macro-label">kcal</div>
                            </div>
                        ` : ''}
                        ${meal.protein_g ? `
                            <div class="macro">
                                <div class="macro-value">${Math.round(meal.protein_g)}g</div>
                                <div class="macro-label">protein</div>
                            </div>
                        ` : ''}
                        ${meal.carbs_g ? `
                            <div class="macro">
                                <div class="macro-value">${Math.round(meal.carbs_g)}g</div>
                                <div class="macro-label">carbs</div>
                            </div>
                        ` : ''}
                        ${meal.fat_g ? `
                            <div class="macro">
                                <div class="macro-value">${Math.round(meal.fat_g)}g</div>
                                <div class="macro-label">fat</div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `).join('');

            // Add delete handlers for meals
            mealsContent.querySelectorAll('.meal-delete').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const card = e.target.closest('.meal-card');
                    const id = card.dataset.id;

                    if (confirm('Delete this meal?')) {
                        const result = await deleteMeal(id);
                        if (result.success) {
                            card.remove();
                            showToast('Meal deleted', 'success');
                            loadStats();
                        } else {
                            showToast('Failed to delete meal', 'error');
                        }
                    }
                });
            });
        }

        function showToast(message, type = 'success') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `
                <span>${type === 'success' ? '✅' : '❌'}</span>
                <span>${message}</span>
            `;
            document.body.appendChild(toast);

            setTimeout(() => toast.remove(), 3000);
        }

        // Load data
        async function loadMemories() {
            memoriesContent.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            try {
                const { memories } = await fetchMemories();
                renderMemories(memories);
            } catch (e) {
                memoriesContent.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Failed to load memories</div></div>';
            }
        }

        async function loadMeals() {
            mealsContent.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            try {
                const { meals } = await fetchMeals();
                renderMeals(meals);
            } catch (e) {
                mealsContent.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Failed to load meals</div></div>';
            }
        }

        async function loadTopics() {
            try {
                const { topics } = await fetchTopics();
                renderTopics(topics);
            } catch (e) {
                topicsCloud.innerHTML = '<div class="empty-state"><p>Failed to load topics</p></div>';
            }
        }

        async function loadStats() {
            try {
                const stats = await fetchStats();
                document.getElementById('stats-memories').textContent = stats.total_memories || 0;
                document.getElementById('stats-meals').textContent = stats.total_meals || 0;
            } catch (e) {}

            // Load graph stats separately
            try {
                const graphStats = await (await fetch('/api/graph/stats')).json();
                document.getElementById('stats-nodes').textContent = graphStats.total_nodes || 0;
            } catch (e) {}
        }

        // Event handlers
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchDebounce);
            searchDebounce = setTimeout(() => {
                searchQuery = e.target.value.trim();
                loadMemories();
            }, 300);
        });

        fromDateInput.addEventListener('change', (e) => {
            fromDate = e.target.value;
            if (currentTab === 'memories') loadMemories();
            else loadMeals();
        });

        toDateInput.addEventListener('change', (e) => {
            toDate = e.target.value;
            if (currentTab === 'memories') loadMemories();
            else loadMeals();
        });

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentTab = tab.dataset.tab;

                memoriesContent.style.display = 'none';
                graphContent.style.display = 'none';
                mealsContent.style.display = 'none';

                // Show/hide sidebar filters based on active tab
                const dateSection = document.getElementById('date-filter-section');
                const topicsSection = document.getElementById('topics-filter-section');

                if (currentTab === 'memories') {
                    memoriesContent.style.display = 'flex';
                    dateSection.style.display = '';
                    topicsSection.style.display = '';
                    loadMemories();
                } else if (currentTab === 'graph') {
                    graphContent.style.display = 'block';
                    dateSection.style.display = 'none';
                    topicsSection.style.display = 'none';
                    initGraph();
                } else {
                    mealsContent.style.display = 'flex';
                    dateSection.style.display = '';
                    topicsSection.style.display = 'none';
                    loadMeals();
                }
            });
        });

        // ─── Graph Explorer ────────────────────────────────────────────
        let graphInitialised = false;
        let graphNodes = [];
        let graphEdges = [];
        let selectedNodeId = null;
        let graphZoom = 1;
        let graphPanX = 0;
        let graphPanY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let hoveredNodeId = null;

        // Layout positions (computed once per data load)
        const nodePositions = new Map();

        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');

        function initGraph() {
            if (!graphInitialised) {
                setupCanvasEvents();
                graphInitialised = true;
            }
            resizeCanvas();
            loadGraphData();
            loadTreeData();
        }

        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth * window.devicePixelRatio;
            canvas.height = container.clientHeight * window.devicePixelRatio;
            canvas.style.width = container.clientWidth + 'px';
            canvas.style.height = container.clientHeight + 'px';
            ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
        }

        async function loadGraphData() {
            try {
                const resp = await fetch('/api/graph/nodes?max_depth=10');
                const data = await resp.json();
                graphNodes = data.nodes || [];
                graphEdges = data.edges || [];
                computeLayout();
                fitToView();
                drawGraph();
            } catch (e) {
                console.error('Failed to load graph:', e);
            }
        }

        async function loadTreeData() {
            const container = document.getElementById('tree-container');
            try {
                const resp = await fetch('/api/graph/tree?max_depth=10');
                const tree = await resp.json();
                container.innerHTML = '';
                if (tree.node) {
                    renderTreeNode(container, tree, 0);
                }
            } catch (e) {
                container.innerHTML = '<div class="detail-empty"><p>Failed to load tree</p></div>';
            }
        }

        function renderTreeNode(container, treeData, depth) {
            const node = treeData.node;
            const children = treeData.children || [];
            const hasChildren = children.length > 0;

            const el = document.createElement('div');

            const nodeEl = document.createElement('div');
            nodeEl.className = 'tree-node' + (selectedNodeId === node.id ? ' selected' : '');
            nodeEl.dataset.nodeId = node.id;
            nodeEl.style.paddingLeft = (0.75 + depth * 0.75) + 'rem';

            const toggle = document.createElement('span');
            toggle.className = 'tree-toggle' + (hasChildren ? ' expanded' : ' leaf');
            toggle.textContent = '▶';

            const nameSpan = document.createElement('span');
            nameSpan.className = 'tree-node-name';
            nameSpan.textContent = node.name;

            nodeEl.appendChild(toggle);
            nodeEl.appendChild(nameSpan);

            if (hasChildren) {
                const countSpan = document.createElement('span');
                countSpan.className = 'tree-node-count';
                countSpan.textContent = children.length;
                nodeEl.appendChild(countSpan);
            }

            nodeEl.addEventListener('click', (e) => {
                e.stopPropagation();
                selectNode(node.id);
            });

            el.appendChild(nodeEl);

            if (hasChildren) {
                const childContainer = document.createElement('div');
                childContainer.className = 'tree-children';

                toggle.addEventListener('click', (e) => {
                    e.stopPropagation();
                    childContainer.classList.toggle('collapsed');
                    toggle.classList.toggle('expanded');
                });

                children.forEach(child => {
                    renderTreeNode(childContainer, child, depth + 1);
                });

                el.appendChild(childContainer);
            }

            container.appendChild(el);
        }

        function computeLayout() {
            nodePositions.clear();
            if (graphNodes.length === 0) return;

            // Build adjacency for tree layout
            const childrenMap = new Map();
            graphNodes.forEach(n => childrenMap.set(n.id, []));
            graphEdges.forEach(e => {
                const list = childrenMap.get(e.source);
                if (list) list.push(e.target);
            });

            // Radial tree layout
            const root = graphNodes.find(n => n.id === 'root') || graphNodes[0];
            const visited = new Set();
            const RING_SPACING = 160;
            const MIN_ARC = 40;

            function layoutSubtree(nodeId, cx, cy, startAngle, endAngle, depth) {
                if (visited.has(nodeId)) return;
                visited.add(nodeId);

                nodePositions.set(nodeId, { x: cx, y: cy });

                const kids = childrenMap.get(nodeId) || [];
                if (kids.length === 0) return;

                const radius = RING_SPACING * (depth + 1);
                const arcPerChild = (endAngle - startAngle) / kids.length;

                kids.forEach((kidId, i) => {
                    const angle = startAngle + arcPerChild * (i + 0.5);
                    const kx = cx + Math.cos(angle) * radius;
                    const ky = cy + Math.sin(angle) * radius;
                    const halfArc = arcPerChild * 0.45;
                    layoutSubtree(kidId, kx, ky, angle - halfArc, angle + halfArc, depth + 1);
                });
            }

            layoutSubtree(root.id, 0, 0, 0, Math.PI * 2, 0);

            // Place any unvisited nodes in a line below
            let offsetX = -200;
            graphNodes.forEach(n => {
                if (!visited.has(n.id)) {
                    nodePositions.set(n.id, { x: offsetX, y: 500 });
                    offsetX += 100;
                }
            });
        }

        function fitToView() {
            if (nodePositions.size === 0) return;

            const cw = canvas.width / window.devicePixelRatio;
            const ch = canvas.height / window.devicePixelRatio;
            const padding = 80;

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            nodePositions.forEach(pos => {
                minX = Math.min(minX, pos.x);
                maxX = Math.max(maxX, pos.x);
                minY = Math.min(minY, pos.y);
                maxY = Math.max(maxY, pos.y);
            });

            const graphW = maxX - minX || 1;
            const graphH = maxY - minY || 1;
            graphZoom = Math.min((cw - padding * 2) / graphW, (ch - padding * 2) / graphH, 2);
            graphZoom = Math.max(graphZoom, 0.1);

            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            graphPanX = cw / 2 - centerX * graphZoom;
            graphPanY = ch / 2 - centerY * graphZoom;

            drawGraph();
        }

        function drawGraph() {
            const cw = canvas.width / window.devicePixelRatio;
            const ch = canvas.height / window.devicePixelRatio;
            ctx.clearRect(0, 0, cw, ch);

            ctx.save();
            ctx.translate(graphPanX, graphPanY);
            ctx.scale(graphZoom, graphZoom);

            // Draw edges
            ctx.lineWidth = 1.5 / graphZoom;
            graphEdges.forEach(edge => {
                const from = nodePositions.get(edge.source);
                const to = nodePositions.get(edge.target);
                if (!from || !to) return;

                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = 'rgba(245, 158, 11, 0.15)';
                ctx.stroke();
            });

            // Draw nodes
            graphNodes.forEach(node => {
                const pos = nodePositions.get(node.id);
                if (!pos) return;

                const isSelected = node.id === selectedNodeId;
                const isHovered = node.id === hoveredNodeId;
                const isRoot = node.id === 'root';
                const baseRadius = isRoot ? 24 : Math.max(12, Math.min(20, 10 + node.access_count * 0.5));
                const radius = baseRadius;

                // Glow for selected/hovered
                if (isSelected || isHovered) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
                    ctx.fillStyle = isSelected
                        ? 'rgba(245, 158, 11, 0.25)'
                        : 'rgba(245, 158, 11, 0.12)';
                    ctx.fill();
                }

                // Node circle
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

                if (isSelected) {
                    ctx.fillStyle = '#f59e0b';
                } else if (isRoot) {
                    ctx.fillStyle = '#1a1d26';
                } else if (node.has_children) {
                    ctx.fillStyle = '#1e222c';
                } else {
                    ctx.fillStyle = '#161920';
                }
                ctx.fill();

                ctx.lineWidth = (isSelected ? 2.5 : 1.5) / graphZoom;
                ctx.strokeStyle = isSelected ? '#fbbf24' : isHovered ? '#f59e0b' : '#27272a';
                ctx.stroke();

                // Label
                const fontSize = Math.max(10, 12 / graphZoom);
                ctx.font = `500 ${fontSize}px Outfit, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = isSelected ? '#0a0b0f' : '#f4f4f5';

                // Truncate name to fit
                let label = node.name;
                if (label.length > 14) label = label.slice(0, 12) + '…';
                ctx.fillText(label, pos.x, pos.y);
            });

            ctx.restore();
        }

        function getNodeAtPosition(screenX, screenY) {
            const x = (screenX - graphPanX) / graphZoom;
            const y = (screenY - graphPanY) / graphZoom;

            for (let i = graphNodes.length - 1; i >= 0; i--) {
                const node = graphNodes[i];
                const pos = nodePositions.get(node.id);
                if (!pos) continue;

                const isRoot = node.id === 'root';
                const radius = isRoot ? 24 : Math.max(12, Math.min(20, 10 + node.access_count * 0.5));
                const dx = pos.x - x;
                const dy = pos.y - y;
                if (dx * dx + dy * dy <= (radius + 4) * (radius + 4)) {
                    return node;
                }
            }
            return null;
        }

        function setupCanvasEvents() {
            canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                dragStartX = e.offsetX;
                dragStartY = e.offsetY;
            });

            canvas.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    graphPanX += e.offsetX - dragStartX;
                    graphPanY += e.offsetY - dragStartY;
                    dragStartX = e.offsetX;
                    dragStartY = e.offsetY;
                    drawGraph();
                } else {
                    const node = getNodeAtPosition(e.offsetX, e.offsetY);
                    const newHovered = node ? node.id : null;
                    if (newHovered !== hoveredNodeId) {
                        hoveredNodeId = newHovered;
                        canvas.style.cursor = newHovered ? 'pointer' : 'grab';
                        drawGraph();
                    }
                }
            });

            canvas.addEventListener('mouseup', (e) => {
                const wasDrag = Math.abs(e.offsetX - dragStartX) > 3 || Math.abs(e.offsetY - dragStartY) > 3;
                isDragging = false;

                if (!wasDrag) {
                    const node = getNodeAtPosition(e.offsetX, e.offsetY);
                    if (node) {
                        selectNode(node.id);
                    }
                }
            });

            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                const mouseX = e.offsetX;
                const mouseY = e.offsetY;

                // Zoom towards mouse position
                graphPanX = mouseX - (mouseX - graphPanX) * delta;
                graphPanY = mouseY - (mouseY - graphPanY) * delta;
                graphZoom *= delta;
                graphZoom = Math.max(0.05, Math.min(5, graphZoom));

                drawGraph();
            }, { passive: false });

            // Toolbar
            document.getElementById('btn-zoom-in').addEventListener('click', () => {
                const cw = canvas.width / window.devicePixelRatio;
                const ch = canvas.height / window.devicePixelRatio;
                graphPanX = cw/2 - (cw/2 - graphPanX) * 1.3;
                graphPanY = ch/2 - (ch/2 - graphPanY) * 1.3;
                graphZoom *= 1.3;
                drawGraph();
            });

            document.getElementById('btn-zoom-out').addEventListener('click', () => {
                const cw = canvas.width / window.devicePixelRatio;
                const ch = canvas.height / window.devicePixelRatio;
                graphPanX = cw/2 - (cw/2 - graphPanX) * 0.7;
                graphPanY = ch/2 - (ch/2 - graphPanY) * 0.7;
                graphZoom *= 0.7;
                drawGraph();
            });

            document.getElementById('btn-fit').addEventListener('click', fitToView);

            document.getElementById('btn-add-node').addEventListener('click', () => {
                showCreateNodeModal(selectedNodeId || 'root');
            });

            document.getElementById('btn-import-diary').addEventListener('click', () => {
                showImportDiaryModal();
            });

            // Resize observer
            new ResizeObserver(() => {
                if (currentTab === 'graph') {
                    resizeCanvas();
                    drawGraph();
                }
            }).observe(canvas.parentElement);
        }

        async function selectNode(nodeId) {
            selectedNodeId = nodeId;

            // Update tree selection highlight in-place (no re-render)
            document.querySelectorAll('.tree-node').forEach(el => {
                el.classList.toggle('selected', el.dataset.nodeId === nodeId);
            });

            // Redraw graph
            drawGraph();

            // Load node details
            const sidebar = document.getElementById('detail-sidebar');
            sidebar.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

            try {
                const resp = await fetch('/api/graph/node/' + nodeId);
                const data = await resp.json();
                renderNodeDetail(data);
            } catch (e) {
                sidebar.innerHTML = '<div class="detail-empty"><p>Failed to load node</p></div>';
            }
        }

        function renderNodeDetail(data) {
            const { node, children, ancestors } = data;
            const sidebar = document.getElementById('detail-sidebar');

            const breadcrumb = ancestors.map((a, i) => {
                const isLast = i === ancestors.length - 1;
                return `<span onclick="selectNode('${a.id}')">${escapeHtml(a.name)}</span>` +
                       (isLast ? '' : '<span class="sep"> › </span>');
            }).join('');

            const childrenHtml = children.length > 0
                ? children.map(c => `
                    <div class="detail-child" onclick="selectNode('${c.id}')">
                        <span>${c.has_children || c.data_token_count > 0 ? '📁' : '📄'}</span>
                        <span class="detail-child-name">${escapeHtml(c.name)}</span>
                        <span class="tree-node-count">${c.data_token_count}t</span>
                    </div>
                `).join('')
                : '<div style="color: var(--text-muted); font-size: 0.85rem;">No children</div>';

            const dataHtml = node.data
                ? `<div class="detail-data">${escapeHtml(node.data)}</div>`
                : '<div style="color: var(--text-muted); font-size: 0.85rem; font-style: italic;">No data stored</div>';

            const lastAccessed = new Date(node.last_accessed).toLocaleDateString('en-GB', {
                day: 'numeric', month: 'short', year: 'numeric'
            });

            sidebar.innerHTML = `
                <div class="detail-breadcrumb">${breadcrumb}</div>
                <div class="detail-name">${escapeHtml(node.name)}</div>
                <div class="detail-description">${escapeHtml(node.description)}</div>

                <div class="detail-meta">
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Accesses</div>
                        <div class="detail-meta-value">${node.access_count}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Tokens</div>
                        <div class="detail-meta-value">${node.data_token_count}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Last seen</div>
                        <div class="detail-meta-value">${lastAccessed}</div>
                    </div>
                    <div class="detail-meta-item">
                        <div class="detail-meta-label">Children</div>
                        <div class="detail-meta-value">${children.length}</div>
                    </div>
                </div>

                <div class="detail-section">
                    <div class="detail-section-title">💾 Data</div>
                    ${dataHtml}
                </div>

                <div class="detail-section">
                    <div class="detail-section-title">📂 Children</div>
                    <div class="detail-children-list">${childrenHtml}</div>
                </div>

                <div class="detail-actions">
                    <button class="detail-action-btn" onclick="editNode('${node.id}')">✏️ Edit</button>
                    <button class="detail-action-btn" onclick="showCreateNodeModal('${node.id}')">➕ Add child</button>
                    ${node.id !== 'root' ? `<button class="detail-action-btn delete" onclick="deleteNode('${node.id}')">🗑️ Delete</button>` : ''}
                </div>
            `;
        }

        async function editNode(nodeId) {
            const resp = await fetch('/api/graph/node/' + nodeId);
            const { node } = await resp.json();

            const sidebar = document.getElementById('detail-sidebar');
            sidebar.innerHTML = `
                <div class="detail-name">✏️ Edit Node</div>
                <div class="modal-field">
                    <label>Name</label>
                    <input type="text" class="detail-edit-field" id="edit-name" value="${escapeHtml(node.name)}" />
                </div>
                <div class="modal-field">
                    <label>Description</label>
                    <textarea class="detail-edit-field" id="edit-desc" rows="3">${escapeHtml(node.description)}</textarea>
                </div>
                <div class="modal-field">
                    <label>Data</label>
                    <textarea class="detail-edit-field" id="edit-data" rows="8">${escapeHtml(node.data)}</textarea>
                </div>
                <div class="detail-actions">
                    <button class="detail-action-btn" onclick="saveNodeEdit('${nodeId}')" style="background: var(--accent-glow); border-color: var(--accent-primary); color: var(--accent-secondary);">💾 Save</button>
                    <button class="detail-action-btn" onclick="selectNode('${nodeId}')">Cancel</button>
                </div>
            `;
        }

        async function saveNodeEdit(nodeId) {
            const name = document.getElementById('edit-name').value.trim();
            const description = document.getElementById('edit-desc').value.trim();
            const data = document.getElementById('edit-data').value;

            if (!name) { showToast('Name is required', 'error'); return; }

            try {
                await fetch('/api/graph/node/' + nodeId, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, description, data })
                });
                showToast('Node updated', 'success');
                loadGraphData();
                loadTreeData();
                selectNode(nodeId);
            } catch (e) {
                showToast('Failed to update', 'error');
            }
        }

        async function deleteNode(nodeId) {
            if (!confirm('Delete this node? Children will be orphaned.')) return;

            try {
                await fetch('/api/graph/node/' + nodeId, { method: 'DELETE' });
                showToast('Node deleted', 'success');
                selectedNodeId = null;
                document.getElementById('detail-sidebar').innerHTML =
                    '<div class="detail-empty"><div class="empty-icon">🧠</div><p>Select a node to view its details</p></div>';
                loadGraphData();
                loadTreeData();
            } catch (e) {
                showToast('Failed to delete', 'error');
            }
        }

        function showCreateNodeModal(parentId) {
            // Remove existing modal if any
            const existing = document.querySelector('.modal-overlay');
            if (existing) existing.remove();

            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.innerHTML = `
                <div class="modal">
                    <h3>✨ New Memory Node</h3>
                    <div class="modal-field">
                        <label>Name</label>
                        <input type="text" class="detail-edit-field" id="new-node-name" placeholder="e.g. Work Projects" />
                    </div>
                    <div class="modal-field">
                        <label>Description</label>
                        <textarea class="detail-edit-field" id="new-node-desc" rows="2" placeholder="Brief description of what this node holds…"></textarea>
                    </div>
                    <div class="modal-field">
                        <label>Data (optional)</label>
                        <textarea class="detail-edit-field" id="new-node-data" rows="4" placeholder="Initial memories…"></textarea>
                    </div>
                    <div class="modal-actions">
                        <button class="modal-btn secondary" onclick="this.closest('.modal-overlay').remove()">Cancel</button>
                        <button class="modal-btn primary" id="btn-create-node">Create</button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);

            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) overlay.remove();
            });

            document.getElementById('btn-create-node').addEventListener('click', async () => {
                const name = document.getElementById('new-node-name').value.trim();
                const description = document.getElementById('new-node-desc').value.trim();
                const data = document.getElementById('new-node-data').value;

                if (!name) { showToast('Name is required', 'error'); return; }

                try {
                    const resp = await fetch('/api/graph/node', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name, description, data, parent_id: parentId })
                    });
                    const result = await resp.json();
                    overlay.remove();
                    showToast('Node created', 'success');
                    loadGraphData();
                    loadTreeData();
                    if (result.node) selectNode(result.node.id);
                } catch (e) {
                    showToast('Failed to create node', 'error');
                }
            });

            document.getElementById('new-node-name').focus();
        }

        function showImportDiaryModal() {
            const existing = document.querySelector('.modal-overlay');
            if (existing) existing.remove();

            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay';
            overlay.innerHTML = `
                <div class="modal">
                    <h3>📥 Import from Diary</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 16px; line-height: 1.5;">
                        Import all existing diary entries into graph memory. Each diary summary
                        will be processed through the LLM to extract facts and organise them
                        into the graph. This may take a while for large diaries.
                    </p>
                    <div id="import-progress" style="display: none;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span id="import-status" style="color: var(--text-secondary); font-size: 0.85em;">Processing…</span>
                            <span id="import-count" style="color: var(--accent-primary); font-size: 0.85em; font-family: 'JetBrains Mono', monospace;">0/0</span>
                        </div>
                        <div style="background: var(--bg-tertiary); border-radius: 6px; height: 8px; overflow: hidden;">
                            <div id="import-bar" style="background: var(--accent-primary); height: 100%; width: 0%; transition: width 0.3s ease; border-radius: 6px;"></div>
                        </div>
                        <div id="import-log" style="margin-top: 12px; max-height: 200px; overflow-y: auto; font-size: 0.8em; font-family: 'JetBrains Mono', monospace; color: var(--text-muted); line-height: 1.6;"></div>
                    </div>
                    <div class="modal-actions" id="import-actions">
                        <button class="modal-btn secondary" onclick="this.closest('.modal-overlay').remove()">Cancel</button>
                        <button class="modal-btn primary" id="btn-start-import">Start Import</button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);

            overlay.addEventListener('click', (e) => {
                if (e.target === overlay && !overlay.dataset.importing) overlay.remove();
            });

            document.getElementById('btn-start-import').addEventListener('click', async () => {
                overlay.dataset.importing = 'true';
                document.getElementById('import-progress').style.display = 'block';
                document.getElementById('btn-start-import').disabled = true;
                document.getElementById('btn-start-import').textContent = 'Importing…';

                try {
                    const resp = await fetch('/api/graph/import-diary', { method: 'POST' });
                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n');
                        buffer = lines.pop();

                        for (const line of lines) {
                            if (!line.trim()) continue;
                            try {
                                const msg = JSON.parse(line);
                                if (msg.type === 'start') {
                                    document.getElementById('import-count').textContent = `0/${msg.total}`;
                                } else if (msg.type === 'progress') {
                                    const pct = Math.round((msg.processed / msg.total) * 100);
                                    document.getElementById('import-bar').style.width = pct + '%';
                                    document.getElementById('import-count').textContent = `${msg.processed}/${msg.total}`;
                                    document.getElementById('import-status').textContent = `Processing ${msg.date}…`;
                                    const log = document.getElementById('import-log');
                                    const icon = msg.error ? '❌' : '📅';
                                    const detail = msg.error ? `error: ${msg.error}` : `${msg.facts} fact${msg.facts !== 1 ? 's' : ''}`;
                                    log.innerHTML += `<div>${icon} ${msg.date} — ${detail}</div>`;
                                    log.scrollTop = log.scrollHeight;
                                } else if (msg.type === 'complete') {
                                    document.getElementById('import-status').textContent = msg.message;
                                    document.getElementById('import-bar').style.width = '100%';
                                    document.getElementById('import-actions').innerHTML = `
                                        <button class="modal-btn primary" onclick="this.closest('.modal-overlay').remove()">Done</button>
                                    `;
                                    delete overlay.dataset.importing;
                                    loadGraphData();
                                    loadTreeData();
                                    showToast('Diary import complete', 'success');
                                } else if (msg.type === 'error') {
                                    document.getElementById('import-status').textContent = 'Error: ' + msg.message;
                                    document.getElementById('import-actions').innerHTML = `
                                        <button class="modal-btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                                    `;
                                    delete overlay.dataset.importing;
                                    showToast('Import failed', 'error');
                                }
                            } catch (e) { /* skip malformed lines */ }
                        }
                    }
                } catch (e) {
                    document.getElementById('import-status').textContent = 'Connection error: ' + e.message;
                    document.getElementById('import-actions').innerHTML = `
                        <button class="modal-btn secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                    `;
                    delete overlay.dataset.importing;
                    showToast('Import failed', 'error');
                }
            });
        }

        // Initial load
        loadStats();
        loadTopics();
        loadMemories();
    </script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the memory viewer server."""
    import sys

    port = 5050
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("\n" + "=" * 60)
    print("🧠 Jarvis Memory Viewer")
    print("=" * 60)
    print(f"\n  📂 Database: {_get_db_path()}")
    print(f"  🌐 URL: http://localhost:{port}")
    print("\n  Press Ctrl+C to stop\n")
    print("=" * 60 + "\n")

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()

