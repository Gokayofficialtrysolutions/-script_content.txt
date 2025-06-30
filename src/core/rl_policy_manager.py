import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from collections import defaultdict
import datetime
import asyncio # For creating task in event handler

if TYPE_CHECKING:
    from src.core.event_system import SystemEvent # For type hinting
    # from src.agents.master_orchestrator import TerminusOrchestrator # Avoid circular import for type hint

PolicyDataType = Dict[str, Dict[str, Dict[str, Any]]]

# Define a structure for storing action preferences
# Q[state_key][action_key] = {"total_reward": float, "count": int, "mean_reward": float, "last_updated_iso": str}
PolicyDataType = Dict[str, Dict[str, Dict[str, Any]]]

class RLPolicyManager:
    """
    Manages Reinforcement Learning policies, specifically action preferences
    based on states, using a simple value averaging (Q-value like) approach.
    Policies are persisted to a JSON file.
    Can optionally listen to events to trigger log processing.
    """
    def __init__(self,
                 policy_storage_path: Path,
                 default_learning_rate: float = 0.1,
                 default_epsilon: float = 0.1,
                 event_bus: Optional[Any] = None, # Accept event_bus, type hint as 'Any' or a Protocol
                 experience_log_file_path: Optional[Path] = None): # Path to the experience log
        """
        Initializes the RLPolicyManager.

        Args:
            policy_storage_path (Path): Path to the JSON file for storing/loading policies.
            default_learning_rate (float): Default learning rate for updates.
            default_epsilon (float): Default epsilon for epsilon-greedy action selection.
            event_bus (Optional[Any]): An event bus instance that supports a
                                       `subscribe_to_event(event_type, handler)` method.
            experience_log_file_path (Optional[Path]): Path to the RL experience log file.
                                                       Required if event_bus is used for auto-processing.
        """
        self.policy_storage_path = policy_storage_path
        self.default_learning_rate = default_learning_rate
        self.default_epsilon = default_epsilon
        self.action_preferences: PolicyDataType = defaultdict(lambda: defaultdict(lambda: {"total_reward": 0.0, "count": 0, "mean_reward": 0.0}))
        self.event_bus = event_bus
        self.experience_log_file_path = experience_log_file_path

        self.load_policy()

        if self.event_bus:
            if self.experience_log_file_path:
                # Type hint for event parameter within the handler
                async def typed_handler(event: 'SystemEvent'): # Use string literal for SystemEvent
                    await self._handle_experience_logged_event(event)

                self.event_bus.subscribe_to_event("rl.experience.logged", typed_handler)
                print(f"RLPolicyManager: Subscribed to 'rl.experience.logged' events. Will process {self.experience_log_file_path}.")
            else:
                print("RLPolicyManager: Warning - Event bus provided but no experience_log_file_path. Cannot auto-process logs via events.")


    async def _handle_experience_logged_event(self, event: 'SystemEvent') -> None:
        """
        Event handler for 'rl.experience.logged' events.
        Triggers processing of the configured experience log file.
        """
        # The event payload might contain the specific log file path if multiple are used,
        # or specific interaction_id for targeted processing.
        # For now, simply re-process the main configured log file.
        print(f"RLPolicyManager: Received '{event.event_type}' event (ID: {event.event_id}). Triggering experience log processing.")
        if self.experience_log_file_path:
            # Run processing in a new task to avoid blocking the event handler if it's called from event loop
            # process_experience_log is synchronous, so wrap with to_thread if it becomes very long,
            # or make process_experience_log itself async (more involved).
            # For now, direct call assuming it's acceptably fast or event loop can handle brief sync call.
            # If process_experience_log is lengthy and sync, it could block event dispatcher.
            # A safer approach for long sync tasks from async handlers:
            # loop = asyncio.get_event_loop()
            # await loop.run_in_executor(None, self.process_experience_log, self.experience_log_file_path)
            # For simplicity now, direct call:
            try:
                # Create a task to avoid blocking the event dispatcher directly if process_experience_log is sync
                # and potentially long-running.
                async def process_log_task():
                    print(f"RLPolicyManager: Starting background task to process log: {self.experience_log_file_path}")
                    self.process_experience_log(self.experience_log_file_path) # Assuming default LR
                    print(f"RLPolicyManager: Background task for log processing completed.")

                asyncio.create_task(process_log_task())

            except Exception as e:
                print(f"RLPolicyManager: Error while trying to process experience log from event: {e}")
        else:
            print("RLPolicyManager: Warning - _handle_experience_logged_event called but no experience_log_file_path configured.")


    def load_policy(self) -> None:
        """
        Loads action preferences from the policy_storage_path.
        If the file doesn't exist, initializes with an empty policy.
        """
        try:
            if self.policy_storage_path.exists():
                with open(self.policy_storage_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # Convert loaded dicts back to defaultdicts with the correct structure
                    self.action_preferences.clear() # Clear existing before loading
                    for state_key, actions in loaded_data.items():
                        self.action_preferences[state_key] = defaultdict(lambda: {"total_reward": 0.0, "count": 0, "mean_reward": 0.0}, actions)
                    print(f"RLPolicyManager: Policy loaded successfully from {self.policy_storage_path}")
            else:
                print(f"RLPolicyManager: Policy file not found at {self.policy_storage_path}. Initializing new policy.")
                # self.action_preferences is already a defaultdict, so it's fine.
        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"RLPolicyManager: Error loading policy from {self.policy_storage_path}: {e}. Using empty policy.")
            self.action_preferences = defaultdict(lambda: defaultdict(lambda: {"total_reward": 0.0, "count": 0, "mean_reward": 0.0}))


    def save_policy(self) -> bool:
        """
        Saves the current action preferences to the policy_storage_path.
        Returns True if successful, False otherwise.
        """
        try:
            self.policy_storage_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert defaultdicts to regular dicts for JSON serialization
            policy_to_save = {sk: dict(av) for sk, av in self.action_preferences.items()}
            with open(self.policy_storage_path, 'w', encoding='utf-8') as f:
                json.dump(policy_to_save, f, indent=2)
            print(f"RLPolicyManager: Policy saved successfully to {self.policy_storage_path}")
            return True
        except IOError as e:
            print(f"RLPolicyManager: Error saving policy to {self.policy_storage_path}: {e}")
            return False

    def update_action_preference(self, state_key: str, action_key: str, reward: float,
                                 learning_rate: Optional[float] = None) -> None:
        """
        Updates the preference for an action in a given state using simple value averaging.
        This is analogous to updating a Q-value in Q-learning with a fixed learning rate of 1/N
        if learning_rate is None, or a constant learning rate if provided.

        Args:
            state_key (str): A string representation of the state.
            action_key (str): A string representation of the action taken.
            reward (float): The reward received for taking the action in the state.
            learning_rate (Optional[float]): If provided, uses this constant learning rate.
                                             If None, uses incremental mean (1/N).
        """
        if not state_key or not action_key:
            print("RLPolicyManager: Warning - state_key or action_key is empty. Skipping update.")
            return

        action_data = self.action_preferences[state_key][action_key]

        current_total_reward = action_data.get("total_reward", 0.0)
        current_count = action_data.get("count", 0)
        current_mean_reward = action_data.get("mean_reward", 0.0)

        new_count = current_count + 1

        if learning_rate is not None: # Constant learning rate
            new_mean_reward = current_mean_reward + learning_rate * (reward - current_mean_reward)
            # total_reward becomes less directly meaningful with a constant learning rate for mean update
            # but we can still track it as a sum of rewards for this (s,a) pair if desired, or adjust its meaning.
            # For simplicity, let's keep it as a sum of rewards.
            new_total_reward = current_total_reward + reward
        else: # Incremental sample average (alpha = 1/N)
            new_total_reward = current_total_reward + reward
            new_mean_reward = new_total_reward / new_count

        self.action_preferences[state_key][action_key] = {
            "total_reward": new_total_reward,
            "count": new_count,
            "mean_reward": new_mean_reward,
            "last_updated_iso": datetime.datetime.utcnow().isoformat()
        }
        # print(f"RLPolicyManager: Updated preference for state='{state_key}', action='{action_key}'. New mean_reward: {new_mean_reward:.4f} (count: {new_count})")


    def get_best_action(self, state_key: str, available_actions: List[str],
                        epsilon: Optional[float] = None) -> Optional[str]:
        """
        Selects an action from available_actions for the given state_key using an epsilon-greedy strategy.

        Args:
            state_key (str): The string representation of the current state.
            available_actions (List[str]): A list of possible action_keys in this state.
            epsilon (Optional[float]): Exploration rate. If None, uses self.default_epsilon.

        Returns:
            Optional[str]: The selected action_key, or None if no actions are available or an error occurs.
        """
        if not available_actions:
            return None

        current_epsilon = epsilon if epsilon is not None else self.default_epsilon

        if random.random() < current_epsilon: # Explore: choose a random action
            selected_action = random.choice(available_actions)
            # print(f"RLPolicyManager: Exploring - randomly selected action '{selected_action}' for state '{state_key}'.")
            return selected_action
        else: # Exploit: choose the best known action
            state_actions = self.action_preferences.get(state_key, {})
            best_action: Optional[str] = None
            max_mean_reward = -float('inf')

            # Find the best action among those available that have been tried
            found_preferred_action = False
            for action_key in available_actions:
                if action_key in state_actions:
                    action_data = state_actions[action_key]
                    if action_data["mean_reward"] > max_mean_reward:
                        max_mean_reward = action_data["mean_reward"]
                        best_action = action_key
                        found_preferred_action = True

            if found_preferred_action and best_action is not None:
                # print(f"RLPolicyManager: Exploiting - selected best action '{best_action}' (mean_reward: {max_mean_reward:.4f}) for state '{state_key}'.")
                return best_action
            else:
                # If no known actions from available_actions, or all have -inf reward (e.g. not tried), pick randomly.
                selected_action = random.choice(available_actions)
                # print(f"RLPolicyManager: Exploiting (no prior preference or all equal) - randomly selected action '{selected_action}' for state '{state_key}'.")
                return selected_action

    def get_action_preferences_for_state(self, state_key: str) -> Dict[str, Dict[str, Any]]:
        """Returns all action preferences for a given state."""
        return dict(self.action_preferences.get(state_key, {}))

    def _construct_state_key(self, state_dict: Dict[str, Any]) -> str:
        """
        Constructs a consistent string key from a state dictionary.
        Sorts by key and joins key-value pairs.
        Example: {"intent": "A", "entities": 2} -> "entities:2;intent:A"
        """
        if not state_dict or not isinstance(state_dict, dict):
            return "default_state_key_if_empty_or_invalid"

        # Filter out complex objects or normalize them if necessary.
        # For now, assuming state_dict contains simple, sortable values.
        # Consider filtering for specific known keys that define the state space.
        relevant_state_keys = sorted([
            "nlu_intent", "nlu_entities_count", "user_prompt_length_category",
            "kb_general_hits", "kg_derived_hits_count", "past_plan_summary_hits_count",
            "kb_plan_log_hits", "kb_feedback_hits", "previous_cycle_outcome"
        ])

        parts = []
        for key in relevant_state_keys:
            if key in state_dict:
                value = state_dict[key]
                # Normalize value to string, handle None
                value_str = str(value) if value is not None else "None"
                parts.append(f"{key}:{value_str}")

        if not parts: # If no relevant keys found, use a generic key or hash of full dict
            # Fallback to hashing the whole dict if no relevant keys are present
            # return "state_fallback_" + str(hash(tuple(sorted(state_dict.items()))))
             return "default_state_key_if_no_relevant_fields"


        return ";".join(parts)

    def process_experience_log(self, log_file_path: Path, learning_rate: Optional[float] = None) -> int:
        """
        Processes an RL experience log file, updating action preferences.

        Args:
            log_file_path (Path): Path to the .jsonl experience log file.
            learning_rate (Optional[float]): If provided, this learning rate will be used for all updates
                                             from this log processing session, overriding the default
                                             or the incremental mean calculation in update_action_preference.

        Returns:
            int: The number of log entries successfully processed.
        """
        if not log_file_path.exists():
            print(f"RLPolicyManager: Experience log file not found at {log_file_path}. No policy updates.")
            return 0

        processed_count = 0
        updated_state_action_pairs = set()

        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        log_entry = json.loads(line.strip())

                        state_dict = log_entry.get("state")
                        action_key = log_entry.get("action_taken") # Renamed from 'action' in logger for clarity
                        reward = log_entry.get("calculated_reward")

                        if state_dict is None or action_key is None or reward is None:
                            print(f"RLPolicyManager: Warning - Skipping log entry at line {line_num} due to missing state, action, or reward.")
                            continue

                        if not isinstance(state_dict, dict):
                            print(f"RLPolicyManager: Warning - Skipping log entry at line {line_num}, 'state' is not a dictionary.")
                            continue
                        if not isinstance(action_key, str):
                             print(f"RLPolicyManager: Warning - Skipping log entry at line {line_num}, 'action_taken' is not a string.")
                             continue
                        if not isinstance(reward, (int, float)):
                            print(f"RLPolicyManager: Warning - Skipping log entry at line {line_num}, 'calculated_reward' is not a number.")
                            continue

                        state_key = self._construct_state_key(state_dict)

                        self.update_action_preference(state_key, action_key, float(reward), learning_rate=learning_rate)
                        updated_state_action_pairs.add((state_key, action_key))
                        processed_count += 1

                    except json.JSONDecodeError:
                        print(f"RLPolicyManager: Warning - Skipping malformed JSON log entry at line {line_num}.")
                    except Exception as e_inner:
                        print(f"RLPolicyManager: Warning - Error processing log entry at line {line_num}: {e_inner}")

            if processed_count > 0:
                self.save_policy()
                print(f"RLPolicyManager: Processed {processed_count} entries from {log_file_path}. Updated {len(updated_state_action_pairs)} unique state-action pairs. Policy saved.")
            else:
                print(f"RLPolicyManager: No valid entries processed from {log_file_path}. Policy not saved.")

            return processed_count

        except IOError as e:
            print(f"RLPolicyManager: Error reading experience log file {log_file_path}: {e}")
            return 0


if __name__ == '__main__':
    # Example Usage
    test_policy_file = Path("./test_rl_policy.json")
    if test_policy_file.exists():
        test_policy_file.unlink() # Clean up before test

    policy_manager = RLPolicyManager(policy_storage_path=test_policy_file, default_epsilon=0.05) # Low epsilon for more exploitation in test

    # Simulate some experiences
    state1 = "intent:code_generation;entities:low"
    action1_s1 = "PlannerStrategy_Simple"
    action2_s1 = "PlannerStrategy_Detailed"

    policy_manager.update_action_preference(state1, action1_s1, reward=1.0)
    policy_manager.update_action_preference(state1, action1_s1, reward=0.5) # Mean should be 0.75
    policy_manager.update_action_preference(state1, action2_s1, reward=0.2) # Mean 0.2

    print(f"\nPreferences for state '{state1}':")
    for action, data in policy_manager.get_action_preferences_for_state(state1).items():
        print(f"  Action: {action}, Data: {data}")

    available_actions_s1 = [action1_s1, action2_s1, "PlannerStrategy_Experimental"]
    print(f"\nGetting best action for state '{state1}' (available: {available_actions_s1}):")
    for _ in range(10): # Show epsilon-greedy behavior
        best = policy_manager.get_best_action(state1, available_actions_s1)
        print(f"  Selected: {best}")

    # Test constant learning rate
    state2 = "intent:web_search"
    action1_s2 = "SearchStrategy_Quick"
    policy_manager.update_action_preference(state2, action1_s2, reward=1.0, learning_rate=0.5) # Mean: 0.0 + 0.5 * (1.0 - 0.0) = 0.5
    policy_manager.update_action_preference(state2, action1_s2, reward=0.0, learning_rate=0.5) # Mean: 0.5 + 0.5 * (0.0 - 0.5) = 0.25
    print(f"\nPreferences for state '{state2}' (LR=0.5):")
    for action, data in policy_manager.get_action_preferences_for_state(state2).items():
        print(f"  Action: {action}, Data: {data}")


    policy_manager.save_policy()

    # Test loading
    print("\nLoading policy into new manager...")
    policy_manager_loaded = RLPolicyManager(policy_storage_path=test_policy_file)
    print(f"Preferences for state '{state1}' (loaded):")
    for action, data in policy_manager_loaded.get_action_preferences_for_state(state1).items():
        print(f"  Action: {action}, Data: {data}")
    assert policy_manager_loaded.action_preferences[state1][action1_s1]["count"] == 2

    if test_policy_file.exists():
        test_policy_file.unlink() # Clean up
        print(f"Cleaned up {test_policy_file}")

    # --- Test process_experience_log ---
    print("\n--- Testing process_experience_log ---")
    dummy_log_file = Path("./dummy_rl_experience.log")
    dummy_policy_file_for_log_test = Path("./test_rl_policy_from_log.json")

    if dummy_policy_file_for_log_test.exists():
        dummy_policy_file_for_log_test.unlink()

    # Create dummy log entries
    log_entries = [
        {
            "rl_interaction_id": "interaction1", "attempt_number": 0, "timestamp_start_iso": "ts1",
            "state": {"nlu_intent": "code_generation", "kb_hits": 2, "previous_cycle_outcome": "none"},
            "action_taken": "Strategy_A", "master_planner_prompt_details": {}, "generated_plan_json": "[]",
            "plan_parsing_status": "success", "final_executed_plan_json": "[]", "execution_status": "success",
            "user_feedback_rating": None, "calculated_reward": 1.0, "next_state": None, "done": True, "timestamp_end_iso": "ts2"
        },
        {
            "rl_interaction_id": "interaction2", "attempt_number": 0, "timestamp_start_iso": "ts3",
            "state": {"nlu_intent": "web_search", "user_prompt_length_category": "short", "previous_cycle_outcome": "success"},
            "action_taken": "Strategy_B", "master_planner_prompt_details": {}, "generated_plan_json": "[]",
            "plan_parsing_status": "success", "final_executed_plan_json": "[]", "execution_status": "failure",
            "user_feedback_rating": "negative", "calculated_reward": -0.5, "next_state": None, "done": True, "timestamp_end_iso": "ts4"
        },
        { # Another for code_generation, Strategy_A to test aggregation
            "rl_interaction_id": "interaction3", "attempt_number": 0, "timestamp_start_iso": "ts5",
            "state": {"nlu_intent": "code_generation", "kb_hits": 2, "previous_cycle_outcome": "none"}, # Same state as first entry
            "action_taken": "Strategy_A", "master_planner_prompt_details": {}, "generated_plan_json": "[]",
            "plan_parsing_status": "success", "final_executed_plan_json": "[]", "execution_status": "success",
            "user_feedback_rating": "positive", "calculated_reward": 1.5, "next_state": None, "done": True, "timestamp_end_iso": "ts6"
        },
        # Malformed entry
        {"broken_json_record": True},
        # Entry with missing critical field
        {
            "rl_interaction_id": "interaction4", "attempt_number": 0, "timestamp_start_iso": "ts7",
            "state": {"nlu_intent": "other"}, # missing action_taken and calculated_reward
            "master_planner_prompt_details": {}, "generated_plan_json": "[]",
            "plan_parsing_status": "success", "final_executed_plan_json": "[]", "execution_status": "success",
            "user_feedback_rating": None, "next_state": None, "done": True, "timestamp_end_iso": "ts8"
        }
    ]
    with open(dummy_log_file, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            if "broken_json_record" in entry:
                f.write("this is not valid json\n")
            else:
                json.dump(entry, f)
                f.write('\n')

    log_test_policy_manager = RLPolicyManager(policy_storage_path=dummy_policy_file_for_log_test)
    processed_count = log_test_policy_manager.process_experience_log(dummy_log_file)
    print(f"Processed {processed_count} entries from dummy log.")

    state_key_cg = log_test_policy_manager._construct_state_key({"nlu_intent": "code_generation", "kb_hits": 2, "previous_cycle_outcome": "none"})
    state_key_ws = log_test_policy_manager._construct_state_key({"nlu_intent": "web_search", "user_prompt_length_category": "short", "previous_cycle_outcome": "success"})

    print(f"\nPreferences after processing log for state '{state_key_cg}':")
    for action, data in log_test_policy_manager.get_action_preferences_for_state(state_key_cg).items():
        print(f"  Action: {action}, Data: {data}")
        if action == "Strategy_A":
            assert data["count"] == 2
            assert data["total_reward"] == 2.5
            assert data["mean_reward"] == 1.25

    print(f"\nPreferences after processing log for state '{state_key_ws}':")
    for action, data in log_test_policy_manager.get_action_preferences_for_state(state_key_ws).items():
        print(f"  Action: {action}, Data: {data}")
        if action == "Strategy_B":
            assert data["count"] == 1
            assert data["total_reward"] == -0.5
            assert data["mean_reward"] == -0.5

    assert processed_count == 3 # 3 valid entries

    if dummy_log_file.exists():
        dummy_log_file.unlink()
        print(f"Cleaned up {dummy_log_file}")
    if dummy_policy_file_for_log_test.exists():
        dummy_policy_file_for_log_test.unlink()
        print(f"Cleaned up {dummy_policy_file_for_log_test}")
```
