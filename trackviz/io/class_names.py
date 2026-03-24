from __future__ import annotations

import re
from typing import List, Optional

# Canonical behavior class names — single source of truth used by both the
# GUI dropdown and the overlay renderer.
BEHAVIOR_NAMES: List[str] = [
    "ProbPumping",    # 0
    "Moving",         # 1
    "Grooming",       # 2
    "Feeding",        # 3
    "Quiescent",      # 4
    "HaltereSwitch",  # 5
    "Twitching",      # 6
    "Defecation",     # 7
]

# Pattern that matches generic auto-generated class names like "cls0", "cls_1",
# "class0", "class 2", etc.
_GENERIC_PATTERN = re.compile(r"^(cls|class)[_\s]?\d+$", re.IGNORECASE)


def resolve_class_names(meta_names: Optional[List[str]]) -> List[str]:
    """Return the best available class name list.

    If *meta_names* is absent or contains only generic placeholder names
    (``cls0``, ``class_1``, …), falls back to :data:`BEHAVIOR_NAMES`.
    Otherwise returns *meta_names* as-is.
    """
    if not meta_names:
        return BEHAVIOR_NAMES
    if all(_GENERIC_PATTERN.match(str(n)) for n in meta_names):
        return BEHAVIOR_NAMES
    return list(meta_names)
