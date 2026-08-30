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
    """
    def __init__(self):
        # to consider adding more signals
        self.override_signals = [
            r"\bactually\b", r"\binstead\b", r"\bchange to\b", 
            r"\bnot\b", r"\brather\b", r"\bnever mind\b", r"\bscratch that\b"
        ]
        # Threshold for over-generality cutoff based on candidate pool size
        self.candidate_cutoff_threshold = 500

    # determines if updated user message contains an intent override signal
    def is_intent_override(self, user_message: str) -> bool:
        message_lower = user_message.lower()
        return any(re.search(pattern, message_lower) for pattern in self.override_signals)

    # checks for over-generality
    def check_over_generality(self, state: ConversationState, candidate_count: Optional[int] = None) -> bool:
        """
        triggers an over-generality cutoff if:
        1. candidate pool overload (exceeds the cutoff threshold).
        2. slot density is too low/only vague details are known (eg., only a high-level category is given).
        """
        # rule 1: candidate pool overload
        if candidate_count is not None and candidate_count > self.candidate_cutoff_threshold:
            return True

        # rule 2: sparse slot check
        has_specific_details = len(state.slots["details"]) > 0
        has_brand_or_price = state.slots["store"] is not None or state.slots["price_max"] is not None
        has_category_only = state.slots["main_category"] is not None and not (has_specific_details or has_brand_or_price)

        if has_category_only or (not has_specific_details and not has_brand_or_price):
            return True

        return False

    # generates clarification prompt upon over-generality detection
    def generate_clarification_prompt(self, state: ConversationState) -> str:
        missing_attributes = []
        if "size" not in state.slots["details"]:
            missing_attributes.append("size")
        if "color" not in state.slots["details"]:
            missing_attributes.append("color")
        if state.slots["store"] is None:
            missing_attributes.append("preferred brand/store")
        if state.slots["price_max"] is None:
            missing_attributes.append("price range")

        category_name = state.slots["main_category"] or "items"
        
        if missing_attributes:
            options_str = ", ".join(missing_attributes[:2])
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