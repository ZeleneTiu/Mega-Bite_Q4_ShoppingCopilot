# to monitor the state of the machine and slot management
# handles information accumulation and intent override
# information accumulation = slot filling, to extract and append information from the user input to the current state of the machine
# intent override = to detect when the user has changed their intent and to modify the state of the machine accordingly (clearing/overwriting slots)

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import re

@dataclass
class ConversationState:
    """
    detects current conversation state and manages slots.
    """
    session_id: str
    current_intent: str = "BROWSING" # default intent = BROWSING
    slots: Dict[str, Any] = field(default_factory=lambda: {
        "parent_asin": None,
        "main_category": None,
        "categories": [],
        "store": None,
        "price_max": None,
        "price_min": None,
        "min_rating": None,
        "details": {}  # stores 'size', 'color', 'materials', etc.
    })
    turn_count: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)

class StateTracker:
    """
    tracks session history, updates item constraints, checks for over-generalizations, and handles intent overrides.
    Uses weighted slot scoring to handle missing prices and prioritize explicit attributes.
    """
    def __init__(self):
        # to consider adding more signals
        self.override_signals = [
            r"\bactually\b", r"\binstead\b", r"\bchange to\b", 
            r"\bnot\b", r"\brather\b", r"\bnever mind\b", r"\bscratch that\b"
        ]
        # Threshold for over-generality cutoff based on candidate pool size
        self.candidate_cutoff_threshold = 500
        
        # Slot weights: higher weight = stronger signal for specificity
        # Explicit attributes weighted equally; price optional
        self.slot_weights = {
            "size": 0.5,           # explicit attribute in details
            "color": 0.5,          # explicit attribute in details
            "material": 0.5,       # explicit attribute in details
            "store": 0.6,          # brand/store signal
            "main_category": 0.3,  # category alone is weaker signal
            "price_max": 0.5,      # price has equal weight when filled
            "price_min": 0.5,      # price has equal weight when filled
            "min_rating": 0.3      # rating is lower priority
        }

    # determines if updated user message contains an intent override signal
    def is_intent_override(self, user_message: str) -> bool:
        message_lower = user_message.lower()
        return any(re.search(pattern, message_lower) for pattern in self.override_signals)

    # checks for over-generality
    def check_over_generality(self, state: ConversationState, candidate_count: Optional[int] = None) -> bool:
        """
        triggers an over-generality cutoff if:
        1. candidate pool overload (exceeds the cutoff threshold).
        2. weighted specificity score is too low (insufficient concrete constraints).
        Price is treated as optional; absence does not penalize specificity.
        """
        # rule 1: candidate pool overload
        if candidate_count is not None and candidate_count > self.candidate_cutoff_threshold:
            return True

        # rule 2: calculate weighted specificity score
        specificity_score = 0.0
        
        # Score details (size, color, material)
        for detail_key in ["size", "color", "material"]:
            if detail_key in state.slots["details"] and state.slots["details"][detail_key] is not None:
                specificity_score += self.slot_weights.get(detail_key, 0.0)
        
        # Score other slots (store, category, rating)
        if state.slots["store"] is not None:
            specificity_score += self.slot_weights["store"]
        if state.slots["main_category"] is not None:
            specificity_score += self.slot_weights["main_category"]
        if state.slots["min_rating"] is not None:
            specificity_score += self.slot_weights["min_rating"]
        
        # Score price slots (optional, equal weight when present)
        if state.slots["price_max"] is not None:
            specificity_score += self.slot_weights["price_max"]
        if state.slots["price_min"] is not None:
            specificity_score += self.slot_weights["price_min"]
        
        # Trigger over-generality if score is too low
        # Minimum threshold: 0.5 (one explicit attribute) OR 0.3 (category + something)
        is_too_generic = specificity_score < 0.5
        
        return is_too_generic

    # generates clarification prompt upon over-generality detection
    def generate_clarification_prompt(self, state: ConversationState) -> str:
        """
        Suggest missing attributes to narrow down results.
        Price is only suggested if the user has already mentioned it in their query history.
        """
        missing_attributes = []
        
        # Always consider explicit attributes first
        if "size" not in state.slots["details"] or state.slots["details"].get("size") is None:
            missing_attributes.append("size")
        if "color" not in state.slots["details"] or state.slots["details"].get("color") is None:
            missing_attributes.append("color")
        if state.slots["store"] is None:
            missing_attributes.append("brand")
        
        # Only suggest price range if user has mentioned price in the conversation
        price_mentioned = any("price" in str(turn.get("content", "")).lower() for turn in state.history)
        if price_mentioned and state.slots["price_max"] is None:
            missing_attributes.append("price range")

        category_name = state.slots["main_category"] or "items"
        
        if missing_attributes:
            # Suggest top 2-3 most relevant missing attributes
            options_str = ", ".join(missing_attributes[:3])
            return f"I found many results for {category_name}! To help me narrow this down, could you specify your {options_str}?"
        
        return f"Could you provide a bit more detail on what kind of {category_name} you're looking for?"

    # updates state based on output from is_intent_override and new constraints detected by intent_router
    def update_state(self, state: ConversationState, user_message: str, new_constraints: Dict[str, Any], intent_track: str) -> ConversationState:
        state.turn_count += 1
        state.current_intent = intent_track
        state.history.append({"role": "user", "content": user_message})

        is_override = self.is_intent_override(user_message)

        # adds specific item details from new_constraints
        details_keys = ["size", "color", "material"]
        for key in list(new_constraints.keys()):
            if key in details_keys:
                if is_override and key in state.slots["details"]:
                    del state.slots["details"][key]
                state.slots["details"][key] = new_constraints.pop(key)

        for key, val in new_constraints.items():
            if is_override and key in state.slots:
                state.slots[key] = None
            if key == "main_category":
                state.slots["main_category"] = val
                if val not in state.slots["categories"]:
                    state.slots["categories"].append(val)
            else:
                state.slots[key] = val

        return state