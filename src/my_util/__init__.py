import json
import re
from typing import Dict, Optional


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """Parse <tag>...</tag> blocks into a dict of tag -> content."""
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


def _find_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of the first JSON object from text."""
    if not text:
        return None
    # Prefer explicit tags when present.
    tags = parse_tags(text)
    if "json" in tags:
        return tags["json"]

    # Try fenced code block first.
    fence_match = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    # Fallback: grab the first {...} block.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None


def extract_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from text, returning None if not found/invalid."""
    candidate = _find_json_block(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
