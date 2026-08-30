# to detect the intent of the user
# classify users into either "buying" or "browsing" intent based on keyword trigger indicators
# "buying" = explicit constraints (eg. brand, item model, item type, price range, etc.)
# "browsing" = implicit constraints (eg. general item type, vague descriptors, etc.)

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

class IntentTrack(str, Enum):
    BUYING = "BUYING"
    BROWSING = "BROWSING"

@dataclass
class IntentResult:
    track: IntentTrack
    confidence: float
    detected_constraints: Dict[str, Any]

class IntentRouter:
    """
    detects intent matching dataset metadata fields:
    (price, store, details, categories, average_rating)
    """
    def __init__(self):
        # high constraint pattern = higher confidence of BUYING intent
        self.constraint_patterns = {
            "size": r"\b(size\s+[xs|s|m|l|xl|xxl|\d.]+)\b",
            "price_max": r"\b(under|below|less than|\$)\s*\$?(\d+(?:\.\d{1,2})?)\b",
            "price_min": r"\b(above|over|more than|at least)\s*\$?(\d+(?:\.\d{1,2})?)\b",
            "store": r"\b(nike|adidas|under armour|levis|zara|puma|calvin klein)\b",
            "color": r"\b(black|white|red|blue|green|yellow|pink|purple|brown|navy|gold|silver)\b",
            "main_category": r"\b(clothing|shoes|jewelry|watches|handbags|boots|sneakers|dresses)\b",
            "min_rating": r"\b(\d(?:\.\d)?)\s*(?:stars?|rated|rating)\b"
        }

        # high prevalence of browsing keywords = higher confidence of BROWSING intent
        self.browsing_keywords = [
            "looking for", "recommend", "ideas", "something", "cozy", 
            "style", "outfit", "casual", "formal", "suggestions", "what goes with", "trending"
        ]

    def route(self, user_message: str) -> IntentResult:
        message_lower = user_message.lower()
        detected_constraints: Dict[str, Any] = {}
        
        # regex extraction
        for constraint_type, pattern in self.constraint_patterns.items():
            match = re.search(pattern, message_lower)
            if match:
                if constraint_type in ("price_max", "price_min"):
                    detected_constraints[constraint_type] = float(match.group(2))
                elif constraint_type == "min_rating":
                    detected_constraints["min_rating"] = float(match.group(1))
                else:
                    detected_constraints[constraint_type] = match.group(1) if match.lastindex else match.group(0)

        has_hard_constraints = len(detected_constraints) > 0
        has_browsing_keywords = any(kw in message_lower for kw in self.browsing_keywords)

        # high constraint density triggers high-precision BUYING track
        if has_hard_constraints and not (has_browsing_keywords and len(detected_constraints) == 1):
            track = IntentTrack.BUYING
            confidence = 0.9 if len(detected_constraints) > 1 else 0.75
        else:
            track = IntentTrack.BROWSING
            confidence = 0.85 if has_browsing_keywords else 0.60

        return IntentResult(
            track=track,
            confidence=confidence,
            detected_constraints=detected_constraints
        )