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


app = Flask(__name__)

# Global database connection
_db_conn: Optional[sqlite3.Connection] = None


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
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
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
                    <span>memories</span>
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
            <div class="sidebar-section">
                <div class="sidebar-title">📅 Date Range</div>
                <div class="date-filters">
                    <input type="date" class="date-input" id="from-date" placeholder="From" />
                    <input type="date" class="date-input" id="to-date" placeholder="To" />
                </div>
            </div>

            <div class="sidebar-section">
                <div class="sidebar-title">🏷️ Topics</div>
                <div class="topics-cloud" id="topics-cloud">
                    <div class="loading"><div class="spinner"></div></div>
                </div>
            </div>
        </aside>

        <section class="content">
            <div class="tabs">
                <button class="tab active" data-tab="memories">
                    <span>💭</span> Memories
                </button>
                <button class="tab" data-tab="meals">
                    <span>🍽️</span> Meals
                </button>
            </div>

            <div id="memories-content" class="memory-list">
                <div class="loading"><div class="spinner"></div></div>
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
        const tabs = document.querySelectorAll('.tab');

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
                        data-topic="${topic.name}">
                    ${topic.name}
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
                    <p class="memory-summary">${memory.summary}</p>
                    ${memory.topics_list.length ? `
                        <div class="memory-topics">
                            ${memory.topics_list.map(t => `<span class="memory-topic">${t}</span>`).join('')}
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
            const { memories } = await fetchMemories();
            renderMemories(memories);
        }

        async function loadMeals() {
            mealsContent.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            const { meals } = await fetchMeals();
            renderMeals(meals);
        }

        async function loadTopics() {
            const { topics } = await fetchTopics();
            renderTopics(topics);
        }

        async function loadStats() {
            const stats = await fetchStats();
            document.getElementById('stats-memories').textContent = stats.total_memories || 0;
            document.getElementById('stats-meals').textContent = stats.total_meals || 0;
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

                if (currentTab === 'memories') {
                    memoriesContent.style.display = 'flex';
                    mealsContent.style.display = 'none';
                    loadMemories();
                } else {
                    memoriesContent.style.display = 'none';
                    mealsContent.style.display = 'flex';
                    loadMeals();
                }
            });
        });

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

