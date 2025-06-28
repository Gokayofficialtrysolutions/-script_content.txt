from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import datetime

@dataclass
class ConversationTurn:
    """
    Represents a single turn in a conversation.
    """
    role: str  # "user", "assistant", "system" (for summaries or directives)
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict) # For plan outcomes, KB IDs, etc.
    keywords: List[str] = field(default_factory=list) # For keyword-based relevance (Phase 2)
    priority_score: float = 0.0 # For prioritizing turns (Phase 2)
    token_count: Optional[int] = None # Estimated token count for this turn's content

    def __post_init__(self):
        # Simple token estimation, can be replaced with a more accurate one.
        # Assumes an average of 4 characters per token.
        if self.token_count is None and self.content:
            self.token_count = len(self.content) // 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "keywords": self.keywords,
            "priority_score": self.priority_score,
            "token_count": self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        # Handle timestamp conversion carefully
        ts_str = data.get("timestamp")
        parsed_timestamp = datetime.datetime.fromisoformat(ts_str) if ts_str else datetime.datetime.utcnow()

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=parsed_timestamp,
            metadata=data.get("metadata", {}),
            keywords=data.get("keywords", []),
            priority_score=data.get("priority_score", 0.0),
            token_count=data.get("token_count") # Will be re-calculated if None by __post_init__
        )

@dataclass
class ContextualHistory:
    """
    Represents the processed, contextually relevant portion of a conversation history,
    ready to be formatted for an LLM prompt.
    """
    turns: List[ConversationTurn] = field(default_factory=list)
    total_turns: int = 0 # Total turns in the original full history
    selected_turns_count: int = 0 # Number of turns included in `turns`
    total_token_estimate: int = 0 # Estimated token count for the selected `turns`
    summary_applied: bool = False # True if summarization was used to generate this context

    def add_turn(self, turn: ConversationTurn, estimate_tokens: bool = True):
        self.turns.append(turn)
        self.selected_turns_count +=1
        if estimate_tokens and turn.token_count is not None:
            self.total_token_estimate += turn.token_count
        elif estimate_tokens and turn.content: # Fallback if token_count wasn't pre-set
             self.total_token_estimate += len(turn.content) // 4


if __name__ == '__main__':
    turn1 = ConversationTurn(role="user", content="Hello there!")
    turn2 = ConversationTurn(role="assistant", content="General Kenobi!",
                             metadata={"is_greeting": True}, token_count=3)

    print(f"Turn 1: {turn1.to_dict()}")
    print(f"Turn 2: {turn2.to_dict()}")
    assert turn1.token_count == len(turn1.content) // 4
    assert turn2.token_count == 3

    hist_dict = turn1.to_dict()
    turn1_restored = ConversationTurn.from_dict(hist_dict)
    assert turn1_restored.role == turn1.role
    assert turn1_restored.content == turn1.content
    assert turn1_restored.timestamp.replace(microsecond=0) == turn1.timestamp.replace(microsecond=0)

    ch = ContextualHistory(total_turns=10)
    ch.add_turn(turn1)
    ch.add_turn(turn2)
    print(f"Contextual History: {ch.selected_turns_count} turns, ~{ch.total_token_estimate} tokens.")
    assert ch.selected_turns_count == 2
    assert ch.total_token_estimate == (len(turn1.content) // 4) + 3

    print("conversation_history.py basic tests passed.")


class ConversationContextManager:
    """
    Manages conversation history to provide contextually relevant portions
    for LLM prompts, considering token limits, recency, relevance, and summarization.
    """
    def __init__(self,
                 full_history: Optional[List[ConversationTurn]] = None,
                 summarization_threshold_turns: int = 20, # Start summarizing if history > this
                 summarization_chunk_size: int = 10,      # How many old turns to summarize at once
                 min_turns_to_keep_raw_at_end: int = 5 # Always keep last N turns unsummarized
                ):
        self._full_history: List[ConversationTurn] = [] # Source of truth, updated externally
        self._managed_history: List[ConversationTurn] = [] # Internal list with summaries, built from _full_history

        self._summarization_threshold_turns = summarization_threshold_turns
        self._summarization_chunk_size = summarization_chunk_size
        # Ensure chunk size is reasonable compared to threshold
        if self._summarization_chunk_size >= self._summarization_threshold_turns:
            self._summarization_chunk_size = max(1, self._summarization_threshold_turns // 2)

        self._min_turns_to_keep_raw_at_end = min_turns_to_keep_raw_at_end

        if full_history:
            self.update_full_history(full_history) # Initialize managed history

    def update_full_history(self, full_history: List[ConversationTurn]):
        """
        Updates the full conversation history and rebuilds the managed_history.
        This effectively resets any existing summaries if the base history changes.
        """
        self._full_history = list(full_history) # Take a copy
        self._managed_history = list(self._full_history) # Rebuild managed history from new full history
        # In a more advanced system, one might try to merge changes and preserve summaries,
        # but for now, a full rebuild of _managed_history upon update is simpler.
        print(f"[ContextManager] Full history updated. Managed history reset to {len(self._managed_history)} turns.")


    def identify_chunk_for_summarization(self) -> Optional[Tuple[List[int], List[ConversationTurn]]]:
        """
        Identifies the oldest chunk of turns in `_managed_history` eligible for summarization.
        Returns a tuple: (list of original indices in _managed_history, list of ConversationTurn objects to summarize).
        Returns None if no chunk needs summarization based on thresholds.
        """
        if len(self._managed_history) < self._summarization_threshold_turns:
            return None # Not long enough to warrant summarization yet

        # Consider turns for summarization, excluding the last `_min_turns_to_keep_raw_at_end`
        # and also excluding any existing summary turns.
        eligible_for_summary_indices = []
        for i, turn in enumerate(self._managed_history[:-self._min_turns_to_keep_raw_at_end]):
            if not (turn.role == "system" and turn.metadata.get("is_summary")):
                eligible_for_summary_indices.append(i)

        if not eligible_for_summary_indices or len(eligible_for_summary_indices) < self._summarization_chunk_size // 2: # Need a decent chunk
            return None

        # Take the oldest `_summarization_chunk_size` from the eligible turns
        # These indices are from the start of `_managed_history`
        indices_to_summarize = eligible_for_summary_indices[:self._summarization_chunk_size]

        if not indices_to_summarize or len(indices_to_summarize) < 2: # Don't summarize a single turn usually
            return None

        turns_to_summarize = [self._managed_history[i] for i in indices_to_summarize]

        print(f"[ContextManager] Identified chunk for summarization: {len(turns_to_summarize)} turns (indices {indices_to_summarize}).")
        return indices_to_summarize, turns_to_summarize

    def replace_turns_with_summary(self, original_indices: List[int], summary_turn: ConversationTurn):
        """
        Replaces turns at the given original_indices in `_managed_history` with a single summary_turn.
        The `original_indices` must be contiguous or handled carefully if not.
        This implementation assumes `original_indices` refers to a block that will be replaced by one summary.
        It's more robust to rebuild the list.
        """
        if not original_indices:
            return

        # Ensure summary_turn has metadata indicating it's a summary
        summary_turn.metadata["is_summary"] = True
        if "original_turn_ids_summarized" not in summary_turn.metadata: # Store IDs if available
            summary_turn.metadata["original_turn_ids_summarized"] = [self._managed_history[i].metadata.get("turn_id", str(i)) for i in original_indices]
        if "original_turn_timestamps" not in summary_turn.metadata:
             summary_turn.metadata["original_turn_timestamps"] = [self._managed_history[i].timestamp.isoformat() for i in original_indices]


        new_managed_history = []
        current_original_idx = 0
        indices_to_replace_set = set(original_indices)

        # Add the summary turn at the position of the first turn it summarizes
        # This assumes original_indices are sorted and represent the block to be replaced.
        first_replacement_index = min(original_indices)

        for i, turn in enumerate(self._managed_history):
            if i == first_replacement_index:
                new_managed_history.append(summary_turn)

            if i not in indices_to_replace_set:
                if i < first_replacement_index or i > max(original_indices) : # if outside the replaced block
                     new_managed_history.append(turn)
                # If inside the block but not the first_replacement_index, it's effectively skipped because
                # the summary turn already covers it.

        self._managed_history = new_managed_history
        print(f"[ContextManager] Replaced {len(original_indices)} turns with summary. New managed history length: {len(self._managed_history)}.")


    def _estimate_token_count(self, text: str) -> int:
        """Roughly estimates token count. Can be replaced with a more precise tokenizer later."""
        if not text:
            return 0
        return len(text) // 4 # Common rough estimate

    def _format_turns_to_prompt_string(self, turns: List[ConversationTurn]) -> str:
        """Formats a list of ConversationTurn objects into a single string for an LLM prompt."""
        # This formatting can be made more sophisticated (e.g., similar to ChatML)
        prompt_lines = []
        for turn in turns:
            prompt_lines.append(f"{turn.role.capitalize()}: {turn.content}")
        return "\n".join(prompt_lines)

    async def get_contextual_history(
        self,
        current_prompt_text: Optional[str] = None, # For future keyword relevance
        max_tokens: int = 3000,
        max_turns: Optional[int] = None, # Overall max turns for the context
        desired_recent_turns: int = 7   # How many of the most recent turns to prioritize
    ) -> ContextualHistory:
        """
        Selects and formats a contextually relevant portion of the conversation history.

        Current Basic Strategy (Phase 1.3):
        - Prioritizes the most recent `desired_recent_turns`.
        - Then, adds more older turns if under `max_tokens` and `max_turns` (if specified),
          preferring more recent ones among those.
        - TODO (Phase 2): Incorporate keyword relevance from `current_prompt_text`.
        - TODO (Phase 3): Incorporate summarization for very long histories.
        """
        contextual_history_obj = ContextualHistory(total_turns=len(self._full_history))

        if not self._full_history:
            return contextual_history_obj # Return empty if no history

        selected_turns: List[ConversationTurn] = []
        current_token_count = 0

        # 1. Add current_prompt_text tokens to current_token_count if it's part of the budget
        # For now, assume max_tokens is for history only, current_prompt is separate.

        if not self._full_history:
            return contextual_history_obj

        selected_turns_dict: Dict[int, ConversationTurn] = {} # Store by original index to maintain order
        current_token_count = 0

        # 1. Prioritize desired_recent_turns (from newest to oldest)
        num_recent_added = 0
        for i in range(len(self._full_history) - 1, -1, -1): # Iterate from newest to oldest
            if num_recent_added >= desired_recent_turns:
                break

            turn = self._full_history[i]
            turn_tokens = turn.token_count if turn.token_count is not None else self._estimate_token_count(turn.content)

            if current_token_count + turn_tokens <= max_tokens:
                if (max_turns is None or len(selected_turns_dict) < max_turns):
                    selected_turns_dict[i] = turn
                    current_token_count += turn_tokens
                    num_recent_added += 1
            else: # Cannot even fit this desired recent turn
                print(f"[ContextManager] WARNING: Desired recent turn (index {i}) could not fit token budget.")
                break

        # 2. Add keyword-relevant older turns if space allows
        # This is a simplified relevance scoring for now.
        # TODO: Make scoring more sophisticated (e.g., TF-IDF, embedding similarity if available)

        relevant_older_turns: List[Tuple[float, int, ConversationTurn]] = [] # (score, original_index, turn)

        # Extract keywords from current prompt if provided
        current_prompt_keywords = set(self.extract_keywords_from_text(current_prompt_text)) if current_prompt_text else set()

        if current_prompt_keywords:
            for i, turn in enumerate(self._full_history):
                if i in selected_turns_dict: # Already selected as a recent turn
                    continue

                # Score based on keyword overlap and built-in priority
                score = turn.priority_score
                turn_keywords = set(turn.keywords) # Assuming keywords are already populated in ConversationTurn
                overlap = len(current_prompt_keywords.intersection(turn_keywords))

                if overlap > 0:
                    score += overlap * 1.0 # Basic overlap score boost

                # Give higher base priority to assistant turns that are plan outcomes
                if turn.role == "assistant" and turn.metadata.get("is_plan_outcome"):
                    score += 0.5

                if score > 0: # Only consider turns with some relevance
                    relevant_older_turns.append((score, i, turn))

            # Sort by score (descending), then by recency (descending index) as a tie-breaker
            relevant_older_turns.sort(key=lambda x: (x[0], x[1]), reverse=True)

            for score, original_idx, turn in relevant_older_turns:
                if max_turns is not None and len(selected_turns_dict) >= max_turns:
                    break

                turn_tokens = turn.token_count if turn.token_count is not None else self._estimate_token_count(turn.content)
                if current_token_count + turn_tokens <= max_tokens:
                    if original_idx not in selected_turns_dict: # Ensure not already added
                        selected_turns_dict[original_idx] = turn
                        current_token_count += turn_tokens
                else: # No more token budget
                    break

        # 3. If still under max_turns and token budget, fill with remaining most recent turns
        # This ensures we try to meet max_turns if possible, after priority selections.
        if max_turns is None or len(selected_turns_dict) < max_turns:
            for i in range(len(self._full_history) - 1, -1, -1):
                if i in selected_turns_dict:
                    continue # Already selected
                if max_turns is not None and len(selected_turns_dict) >= max_turns:
                    break

                turn = self._full_history[i]
                turn_tokens = turn.token_count if turn.token_count is not None else self._estimate_token_count(turn.content)
                if current_token_count + turn_tokens <= max_tokens:
                    selected_turns_dict[i] = turn
                    current_token_count += turn_tokens
                else:
                    # This turn doesn't fit, and since we are iterating from recent, older ones won't either.
                    break

        # Assemble the final list of turns in chronological order
        final_selected_indices = sorted(selected_turns_dict.keys())
        final_turns = [selected_turns_dict[idx] for idx in final_selected_indices]

        for t in final_turns: # Add to the ContextualHistory object
            contextual_history_obj.add_turn(t, estimate_tokens=False) # Tokens already accounted for
        contextual_history_obj.total_token_estimate = current_token_count

        return contextual_history_obj

    def format_history_for_prompt(self, contextual_history: ContextualHistory) -> str:
        """
        Takes a ContextualHistory object and returns a single formatted string.
        """
        return self._format_turns_to_prompt_string(contextual_history.turns)

    def extract_keywords_from_text(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Simple keyword extraction from text.
        - Lowercases
        - Removes common stop words
        - Takes words longer than 3 chars
        - TODO: Add quoted phrase extraction
        """
        if not text:
            return []

        # Basic English stop words. This list can be expanded.
        stop_words = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
            "could", "may", "might", "must", "and", "but", "or", "nor", "for", "so", "yet",
            "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "mine", "yours", "hers", "ours", "theirs", "to", "of", "in", "on", "at", "by",
            "from", "with", "about", "above", "after", "again", "against", "all", "any",
            "because", "before", "below", "between", "both", "during", "each", "few", "further",
            "here", "into", "just", "no", "not", "now", "once", "only", "other", "out", "over",
            "own", "same", "some", "such", "than", "that's", "too", "under", "until", "up",
            "very", "while", "through", "tell", "me", "please", "give", "provide", "about", "make"
        ])

        text_lower = text.lower()
        # Remove punctuation (basic) - can be improved with regex for more complex cases
        text_no_punct = ''.join(char for char in text_lower if char.isalnum() or char.isspace())

        words = text_no_punct.split()

        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 3 and not word.isdigit()
        ]

        # Simple frequency count to get top keywords if many are extracted
        if len(keywords) > max_keywords:
            from collections import Counter
            keyword_counts = Counter(keywords)
            top_keywords = [kw for kw, count in keyword_counts.most_common(max_keywords)]
            return top_keywords

        return list(set(keywords)) # Return unique keywords


if __name__ == '__main__':
    # ... (previous tests for ConversationTurn and ContextualHistory) ...
    print("\n--- ConversationContextManager Tests ---")

    # Sample history
    history_turns = [
        ConversationTurn(role="user", content="What's the weather like?"), # ~5 tokens
        ConversationTurn(role="assistant", content="It's sunny in California."), # ~5 tokens
        ConversationTurn(role="user", content="Tell me about large language models."), # ~7 tokens
        ConversationTurn(role="assistant", content="LLMs are powerful AI... (long explanation)"), # Assume 100 tokens
        ConversationTurn(role="user", content="Thanks! Now, what about a good place for dinner?"), # ~10 tokens
        ConversationTurn(role="assistant", content="Italiano's is great for pasta."), # ~6 tokens
        ConversationTurn(role="user", content="How do I get there?"), # ~5 tokens
        ConversationTurn(role="assistant", content="Take Main Street then a left on Oak."), # ~8 tokens
        ConversationTurn(role="user", content="Perfect, thanks!"), # ~3 tokens
    ]
    for t in history_turns: t.__post_init__() # Ensure token_counts are set

    manager = ConversationContextManager(full_history=history_turns)

    # Test 1: Basic recency
    ctx_hist1 = asyncio.run(manager.get_contextual_history(max_tokens=50, desired_recent_turns=3))
    formatted_str1 = manager.format_history_for_prompt(ctx_hist1)
    print(f"\nTest 1 (max_tokens=50, desired_recent=3): {ctx_hist1.selected_turns_count} turns, ~{ctx_hist1.total_token_estimate} tokens")
    print(formatted_str1)
    # Expected: Last few turns that fit under 50 tokens.
    # "Take Main Street..." (8) + "Perfect, thanks!" (3) = 11
    # "Italiano's is great..." (6) + 11 = 17
    # "How do I get there?" (5) + 17 = 22.  These are the 3 most recent.
    # "Thanks! Now, what about..." (10) + 22 = 32.
    # "LLMs are powerful..." (100) - too large.
    # So should get last 4 turns.
    assert ctx_hist1.selected_turns_count >= 3 # Should try to get at least desired_recent_turns if they fit
    assert ctx_hist1.total_token_estimate <= 50

    # Test 2: Max turns limit
    ctx_hist2 = asyncio.run(manager.get_contextual_history(max_tokens=200, max_turns=2, desired_recent_turns=5))
    formatted_str2 = manager.format_history_for_prompt(ctx_hist2)
    print(f"\nTest 2 (max_tokens=200, max_turns=2, desired_recent=5): {ctx_hist2.selected_turns_count} turns, ~{ctx_hist2.total_token_estimate} tokens")
    print(formatted_str2)
    assert ctx_hist2.selected_turns_count == 2 # Limited by max_turns

    # Test 3: Very small token limit
    ctx_hist3 = asyncio.run(manager.get_contextual_history(max_tokens=10, desired_recent_turns=3))
    formatted_str3 = manager.format_history_for_prompt(ctx_hist3)
    print(f"\nTest 3 (max_tokens=10, desired_recent=3): {ctx_hist3.selected_turns_count} turns, ~{ctx_hist3.total_token_estimate} tokens")
    print(formatted_str3)
    # "Perfect, thanks!" (3)
    # "Take Main Street..." (8) - this alone is 8. So only this and "Perfect, thanks!" might not fit if prev was also big.
    # Ah, the loop adds newest first then reverses.
    # "Perfect, thanks!" (3 tokens) - added. current_tokens = 3
    # "Take Main Street..." (8 tokens) - 3 + 8 = 11. Exceeds 10. So only "Perfect, thanks!"
    assert ctx_hist3.selected_turns_count == 1
    assert ctx_hist3.turns[0].content == "Perfect, thanks!"

    # Test 4: Empty history
    empty_manager = ConversationContextManager()
    ctx_hist4 = asyncio.run(empty_manager.get_contextual_history())
    assert ctx_hist4.selected_turns_count == 0
    assert ctx_hist4.total_token_estimate == 0
    print("\nTest 4 (empty history) passed.")

    print("\nConversationContextManager tests passed.")
