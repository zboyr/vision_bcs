"""Response parsing for different output modes."""

import json
import re
from typing import Any, Dict, Optional


def parse_integer(content: str) -> Optional[int]:
    """Parse a single integer BCS (1-9) from response text."""
    text = content.strip()
    if re.fullmatch(r"[1-9]", text):
        return int(text)
    m = re.search(r"\b([1-9])\b", text)
    return int(m.group(1)) if m else None


def parse_json_bcs(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response containing a BCS score.

    Looks for ``bcs`` or ``bcs_primary`` key in a JSON object.
    Returns dict with ``bcs`` (int), ``reasoning`` (str), ``confidence``,
    ``raw_json`` (full parsed dict), or *None* on failure.
    """
    for candidate in _extract_json_candidates(content):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        bcs_val = data.get("bcs") or data.get("bcs_primary")
        if bcs_val is None:
            continue
        try:
            bcs = int(bcs_val)
        except (ValueError, TypeError):
            continue
        if 1 <= bcs <= 9:
            return {
                "bcs": bcs,
                "reasoning": str(data.get("reasoning", "")),
                "confidence": data.get("confidence"),
                "raw_json": data,
            }
    return None


def _extract_json_candidates(content: str) -> list[str]:
    """Yield possible JSON substrings from *content*."""
    candidates = [content.strip()]
    # Markdown fenced block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())
    # Outermost { … }
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    # Innermost (non-nested) { … }
    m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    return candidates


def extract_bcs(content: str) -> Optional[int]:
    """Best-effort extraction: try JSON first, then bare integer."""
    result = parse_json_bcs(content)
    if result:
        return result["bcs"]
    return parse_integer(content)
