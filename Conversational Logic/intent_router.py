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
    Uses weighted scoring to handle missing prices and prioritize explicit constraints.
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

        # Constraint weights: higher weight = stronger signal for BUYING intent
        # Explicit constraints (specific attributes) weighted higher than price
        self.constraint_weights = {
            "size": 0.5,           # explicit attribute
            "color": 0.5,          # explicit attribute
            "store": 0.6,          # explicit brand/store signal
            "main_category": 0.3,  # category alone is weaker signal
            "price_max": 0.5,      # price has equal weight when mentioned
            "price_min": 0.5,      # price has equal weight when mentioned
            "min_rating": 0.3      # rating is lower priority
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

        has_browsing_keywords = any(kw in message_lower for kw in self.browsing_keywords)

        # Calculate weighted constraint score (regardless of price availability in dataset)
        constraint_score = sum(
            self.constraint_weights.get(constraint_type, 0.0)
            for constraint_type in detected_constraints.keys()
        )
        
        # Count explicit constraints (non-price, non-category attributes)
        explicit_constraints = [
            c for c in detected_constraints.keys() 
            if c in ("size", "color", "store")
        ]
        has_explicit_constraints = len(explicit_constraints) > 0

        # Determine intent and confidence based on weighted score
        # If user has explicit attributes (size, color, brand) OR
        # has price constraints AND not purely browsing keywords -> BUYING
        if has_explicit_constraints or (constraint_score >= 0.7 and not has_browsing_keywords):
            track = IntentTrack.BUYING
            # Higher confidence for multiple explicit constraints
            if len(explicit_constraints) >= 2:
                confidence = 0.95
            elif has_explicit_constraints:
                confidence = 0.85
            else:
                confidence = 0.75
        elif has_browsing_keywords and constraint_score < 0.4:
            # Clear browsing intent with minimal specific constraints
            track = IntentTrack.BROWSING
            confidence = 0.85
        elif constraint_score >= 0.4:
            # Some constraints detected but mixed signals
            track = IntentTrack.BUYING
            confidence = 0.70
        else:
            # No clear constraints, default to browsing
            track = IntentTrack.BROWSING
            confidence = 0.60

        return IntentResult(
            track=track,
            confidence=confidence,
            detected_constraints=detected_constraints
        )