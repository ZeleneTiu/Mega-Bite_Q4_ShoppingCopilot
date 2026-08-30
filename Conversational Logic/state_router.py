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
        2. dynamic weighted specificity score is too low (insufficient concrete constraints).
        Price is treated as optional; absence does not penalize specificity.
        """
        # rule 1: candidate pool overload
        if candidate_count is not None and candidate_count > self.candidate_cutoff_threshold:
            return True

        # rule 2: calculate dynamic weighted specificity score based on filled slots
        specificity_score = self._calculate_dynamic_specificity_score(state)
        
        # Trigger over-generality if score is too low
        # Minimum threshold: 0.5 (one explicit attribute) OR 0.3 (category + something)
        is_too_generic = specificity_score < 0.5
        
        return is_too_generic
    
    def _calculate_dynamic_specificity_score(self, state: ConversationState) -> float:
        """
        Calculate specificity score with dynamic weighting.
        Weights adjust based on what's already filled to prioritize most discriminative attributes.
        """
        specificity_score = 0.0
        
        # Determine what's filled
        has_category = state.slots["main_category"] is not None
        has_store = state.slots["store"] is not None
        has_size = "size" in state.slots["details"] and state.slots["details"]["size"] is not None
        has_color = "color" in state.slots["details"] and state.slots["details"].get("color") is not None
        has_price = state.slots["price_max"] is not None or state.slots["price_min"] is not None
        
        # Dynamic weighting based on state
        # If category is missing, it's critical; heavily penalize
        if not has_category:
            return 0.0
        
        # Category established: base score
        specificity_score = 0.3
        
        # Store/brand is highly discriminative
        if has_store:
            specificity_score += 0.3
        
        # Details (size, color) are very valuable when category + store known
        if has_size:
            specificity_score += 0.2
        if has_color:
            specificity_score += 0.2
        
        # Price adds minimal value unless already have category + store
        if has_price and (has_size or has_color or has_store):
            specificity_score += 0.1
        
        return specificity_score

    # generates clarification prompt upon over-generality detection
    def generate_clarification_prompt(self, state: ConversationState) -> str:
        """
        Suggest missing attributes to narrow down results.
        Uses dynamic prioritization and caps at 2 attributes max.
        Price is only suggested if the user has already mentioned it in their query history.
        """
        missing_attributes = self._get_prioritized_missing_attributes(state)
        category_name = state.slots["main_category"] or "items"
        
        if missing_attributes:
            # Cap at 2 attributes to avoid overwhelming the user
            options_str = ", ".join(missing_attributes[:2])
            return f"I found many results for {category_name}! To help me narrow this down, could you specify your {options_str}?"
        
        return f"Could you provide a bit more detail on what kind of {category_name} you're looking for?"
    
    def _get_prioritized_missing_attributes(self, state: ConversationState) -> List[str]:
        """
        Get missing attributes sorted by priority based on what's already filled.
        Prioritizes attributes that are most discriminative given current state.
        """
        missing_attributes = []
        
        # Build priority list based on filled slots
        has_category = state.slots["main_category"] is not None
        has_store = state.slots["store"] is not None
        has_size = "size" in state.slots["details"] and state.slots["details"].get("size") is not None
        has_color = "color" in state.slots["details"] and state.slots["details"].get("color") is not None
        has_price = state.slots["price_max"] is not None or state.slots["price_min"] is not None
        
        filled_count = sum([has_category, has_store, has_size, has_color, has_price])
        
        # Dynamic prioritization: ask for most discriminative attributes first
        if not has_category:
            # Category is the foundation, ask first if missing
            missing_attributes.append("category")
        elif not has_store:
            # Once category known, brand/store is highly discriminative
            missing_attributes.append("brand")
        elif not has_size and not has_color:
            # With category + store, size and color are most discriminative
            missing_attributes.append("size")
            missing_attributes.append("color")
        elif not has_size:
            missing_attributes.append("size")
        elif not has_color:
            missing_attributes.append("color")
        
        # Only suggest price if user has mentioned it before
        price_mentioned = any("price" in str(turn.get("content", "")).lower() for turn in state.history)
        if price_mentioned and not has_price:
            missing_attributes.append("price range")
        
        return missing_attributes

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