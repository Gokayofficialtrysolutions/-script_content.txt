import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
import datetime

# Define a structure for storing action preferences
# Q[state_key][action_key] = {"total_reward": float, "count": int, "mean_reward": float, "last_updated_iso": str}
PolicyDataType = Dict[str, Dict[str, Dict[str, Any]]]

class RLPolicyManager:
    """
    Manages Reinforcement Learning policies, specifically action preferences
    based on states, using a simple value averaging (Q-value like) approach.
    Policies are persisted to a JSON file.
    """
    def __init__(self, policy_storage_path: Path, default_learning_rate: float = 0.1, default_epsilon: float = 0.1):
        """
        Initializes the RLPolicyManager.

        Args:
            policy_storage_path (Path): Path to the JSON file for storing/loading policies.
            default_learning_rate (float): Default learning rate for updates if not specified.
            default_epsilon (float): Default epsilon for epsilon-greedy action selection.
                                     Probability of choosing a random action (exploration).
        """
        self.policy_storage_path = policy_storage_path
        self.default_learning_rate = default_learning_rate
        self.default_epsilon = default_epsilon
        self.action_preferences: PolicyDataType = defaultdict(lambda: defaultdict(lambda: {"total_reward": 0.0, "count": 0, "mean_reward": 0.0}))
        self.load_policy()

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
        print(f"\nCleaned up {test_policy_file}")

```
