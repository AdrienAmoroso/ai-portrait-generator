"""Prompt construction with sanitization and per-player variety injection."""

import hashlib
from dataclasses import dataclass

from config import GLOBAL_PROMPT, NEGATIVE_PROMPT

# Variety pools — deterministically selected per player via ID hash

LIGHTING_VARIANTS = [
    "soft diffused lighting from left",
    "warm golden hour side lighting",
    "cool studio fill lighting",
    "natural daylight with gentle rim light",
    "dramatic Rembrandt lighting",
    "bright even studio lighting",
    "soft window light from right",
]

EXPRESSION_VARIANTS = [
    "confident gaze",
    "relaxed natural expression",
    "slight smile",
    "focused determined look",
    "warm approachable expression",
    "calm composed expression",
    "subtle intensity in the eyes",
]

# Forbidden words removed from prompts (brands, equipment, etc.)
_FORBIDDEN_WORDS = {
    "tennis ball", "racket", "racquet", "nike", "adidas", "wilson",
    "head brand", "lacoste", "fila", "under armour", "yonex",
    "babolat", "backwards", "worn backwards",
    "puma", "asics", "new balance", "uniqlo", "lotto", "diadora",
    "reebok", "champion", "kappa", "mizuno", "ellesse", "hoka",
    "on running", "lululemon", "k-swiss", "prince", "dunlop",
    "tecnifibre", "artengo", "le coq sportif", "sergio tacchini",
    "swoosh", "three stripes", "logo", "branded", "sponsor",
}


@dataclass
class BuiltPrompt:
    """A fully constructed prompt ready for generation."""
    positive: str
    negative: str


def _pick_variant(pool: list[str], player_id: int, salt: str = "") -> str:
    """Deterministically pick one variant from a pool based on player ID."""
    digest = hashlib.md5(
        f"{player_id}:{salt}".encode(), usedforsecurity=False
    ).hexdigest()
    index = int(digest, 16) % len(pool)
    return pool[index]


def _sanitize_prompt(text: str) -> str:
    """Remove forbidden words / brand references from a prompt string."""
    result = text
    for word in _FORBIDDEN_WORDS:
        # Case-insensitive removal
        lower = result.lower()
        pos = lower.find(word)
        while pos != -1:
            result = result[:pos] + result[pos + len(word):]
            lower = result.lower()
            pos = lower.find(word)
    # Clean up double commas / spaces left behind
    while ",," in result:
        result = result.replace(",,", ",")
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip().strip(",").strip()


def _replace_tennis_clothing(text: str) -> str:
    """Replace 'tennis' in clothing descriptions with 'athletic'."""
    import re
    # Replace "tennis" when it precedes clothing words
    clothing_pattern = re.compile(
        r"\btennis\s+(tee\s+shirt|t-shirt|shirt|polo|dress|tank\s+top|top|"
        r"shorts|skirt|cap|hat|visor|headband)",
        re.IGNORECASE,
    )
    return clothing_pattern.sub(r"athletic \1", text)


def build_prompt(
    player_id: int,
    player_prompt: str,
    player_details: str = "",
) -> BuiltPrompt:
    """Build the full positive and negative prompts for a player.

    Applies:
    - global prompt prefix
    - prompt sanitization (brand removal, tennis → athletic)
    - variety injection (lighting, expression)
    - negative prompt
    """
    # Combine and sanitize the player-specific text
    parts = [player_prompt, player_details]
    raw_player = " ".join(p for p in parts if p)
    clean_player = _sanitize_prompt(raw_player)
    clean_player = _replace_tennis_clothing(clean_player)

    # Handle backward caps → forward caps
    import re
    clean_player = re.sub(
        r"cap\s+(made\s+of\s+.+?\s+)?worn\s+backwards",
        r"cap \1worn forward",
        clean_player,
        flags=re.IGNORECASE,
    )
    clean_player = re.sub(
        r"worn\s+backwards",
        "worn forward",
        clean_player,
        flags=re.IGNORECASE,
    )

    # Pick variety modifiers deterministically
    lighting = _pick_variant(LIGHTING_VARIANTS, player_id, "light")
    expression = _pick_variant(EXPRESSION_VARIANTS, player_id, "expr")

    # Build the final positive prompt
    positive = ", ".join(
        filter(None, [
            GLOBAL_PROMPT,
            clean_player,
            lighting,
            expression,
        ])
    )

    return BuiltPrompt(positive=positive, negative=NEGATIVE_PROMPT)
