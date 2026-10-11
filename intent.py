"""
intent.py
─────────
Domain-agnostic routing helper for the RAG Engine.
Determines whether a user query should trigger RAG retrieval.
Supports dynamic keyword loading from domain configuration or defaults to open RAG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config path relative to repository root
_CONFIG_PATH = Path(__file__).resolve().parent / "domain_config.json"


def load_domain_keywords() -> set[str]:
    """Load dynamic routing keywords from domain configuration if available."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return set(data.get("routing", {}).get("intent_keywords", []))
        except Exception as exc:
            logger.warning("[INTENT] Failed to load domain config: %s", exc)
    return set()


_ALWAYS_USE_RAG: bool = True
_CUSTOM_KEYWORDS: set[str] = load_domain_keywords()


def should_use_rag(query: str) -> bool:
    """
    Return True if the query should trigger RAG.
    Default behaviour is open (True for all queries).
    If custom intent keywords are configured, checks substring match.
    """
    if _ALWAYS_USE_RAG or not _CUSTOM_KEYWORDS:
        return True

    q = query.lower().strip()
    return any(kw in q for kw in _CUSTOM_KEYWORDS)

