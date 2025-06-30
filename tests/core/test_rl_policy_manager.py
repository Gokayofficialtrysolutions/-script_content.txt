"""
Unit tests for the RLPolicyManager class, focusing on its ability to
construct state keys and process experience log files to update action preferences.
"""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import datetime # For checking last_updated_iso if needed, though not primary focus here

from src.core.rl_policy_manager import RLPolicyManager

@pytest.fixture
def temp_policy_file(tmp_path: Path) -> Path:
    """Provides a temporary file path for policy storage."""
    return tmp_path / "test_policy.json"

@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Provides a temporary file path for dummy log files."""
    return tmp_path / "dummy_experience.log"

@pytest.fixture
def policy_manager_instance(temp_policy_file: Path) -> RLPolicyManager:
    """Fixture to create an RLPolicyManager instance with a temporary policy file."""
    # Ensure the policy file does not exist from a previous test run if tests are not isolated by tmp_path
    if temp_policy_file.exists():
        temp_policy_file.unlink()
    return RLPolicyManager(policy_storage_path=temp_policy_file)

class TestRLPolicyManagerConstructStateKey:
    def test_construct_state_key_valid_dict(self, policy_manager_instance: RLPolicyManager):
        state_dict = {
            "nlu_intent": "code_generation",
            "kb_feedback_hits": 1,
            "user_prompt_length_category": "medium",
            "some_other_field": "ignore_this" # Should be ignored
        }
        # Expected key from sorted relevant keys:
        # kb_feedback_hits:1;nlu_intent:code_generation;user_prompt_length_category:medium
        # The relevant_state_keys list in RLPolicyManager is:
        # ["nlu_intent", "nlu_entities_count", "user_prompt_length_category",
        #  "kb_general_hits", "kg_derived_hits_count", "past_plan_summary_hits_count",
        #  "kb_plan_log_hits", "kb_feedback_hits", "previous_cycle_outcome"]
        expected_key = "kb_feedback_hits:1;nlu_intent:code_generation;user_prompt_length_category:medium"
        assert policy_manager_instance._construct_state_key(state_dict) == expected_key

    def test_construct_state_key_all_relevant_keys(self, policy_manager_instance: RLPolicyManager):
        state_dict = {
            "nlu_intent": "A", "nlu_entities_count": 2, "user_prompt_length_category": "short",
            "kb_general_hits": 1, "kg_derived_hits_count": 0, "past_plan_summary_hits_count": 1,
            "kb_plan_log_hits": 3, "kb_feedback_hits": 0, "previous_cycle_outcome": "success"
        }
        # Order will be alphabetical by key as per relevant_state_keys sorting
        expected_parts = [
            "kb_feedback_hits:0", "kb_general_hits:1", "kb_plan_log_hits:3",
            "kg_derived_hits_count:0", "nlu_entities_count:2", "nlu_intent:A",
            "past_plan_summary_hits_count:1", "previous_cycle_outcome:success",
            "user_prompt_length_category:short"
        ]
        expected_key = ";".join(expected_parts)
        assert policy_manager_instance._construct_state_key(state_dict) == expected_key


    def test_construct_state_key_some_relevant_keys_missing(self, policy_manager_instance: RLPolicyManager):
        state_dict = {"nlu_intent": "web_search", "previous_cycle_outcome": "failure"}
        expected_key = "nlu_intent:web_search;previous_cycle_outcome:failure"
        assert policy_manager_instance._construct_state_key(state_dict) == expected_key

    def test_construct_state_key_handles_none_values(self, policy_manager_instance: RLPolicyManager):
        state_dict = {"nlu_intent": "query", "kb_general_hits": None}
        expected_key = "kb_general_hits:None;nlu_intent:query"
        assert policy_manager_instance._construct_state_key(state_dict) == expected_key

    def test_construct_state_key_empty_dict(self, policy_manager_instance: RLPolicyManager):
        assert policy_manager_instance._construct_state_key({}) == "default_state_key_if_no_relevant_fields"

    def test_construct_state_key_no_relevant_keys(self, policy_manager_instance: RLPolicyManager):
        state_dict = {"non_relevant_key": "value", "another_one": 123}
        assert policy_manager_instance._construct_state_key(state_dict) == "default_state_key_if_no_relevant_fields"

    def test_construct_state_key_invalid_input(self, policy_manager_instance: RLPolicyManager):
        assert policy_manager_instance._construct_state_key(None) == "default_state_key_if_empty_or_invalid" # type: ignore
        assert policy_manager_instance._construct_state_key("not_a_dict") == "default_state_key_if_empty_or_invalid" # type: ignore

@pytest.mark.asyncio # Though methods are sync, fixture might be async in future or for consistency
class TestRLPolicyManagerProcessExperienceLog:

    def test_process_empty_log_file(self, policy_manager_instance: RLPolicyManager, temp_log_file: Path):
        temp_log_file.touch() # Create empty file

        # Spy on update_action_preference and save_policy
        policy_manager_instance.update_action_preference = MagicMock()
        policy_manager_instance.save_policy = MagicMock()

        processed_count = policy_manager_instance.process_experience_log(temp_log_file)

        assert processed_count == 0
        policy_manager_instance.update_action_preference.assert_not_called()
        policy_manager_instance.save_policy.assert_not_called() # Should not save if nothing processed

    def test_process_log_file_not_found(self, policy_manager_instance: RLPolicyManager):
        non_existent_log_file = Path("./non_existent_dummy.log")
        if non_existent_log_file.exists(): non_existent_log_file.unlink()

        processed_count = policy_manager_instance.process_experience_log(non_existent_log_file)
        assert processed_count == 0

    def test_process_valid_log_entries(self, policy_manager_instance: RLPolicyManager, temp_log_file: Path):
        log_entries_data = [
            {"state": {"nlu_intent": "A"}, "action_taken": "Action1", "calculated_reward": 1.0},
            {"state": {"nlu_intent": "B"}, "action_taken": "Action2", "calculated_reward": -0.5},
            {"state": {"nlu_intent": "A"}, "action_taken": "Action1", "calculated_reward": 0.5}, # Repeat state-action
        ]
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            for entry in log_entries_data:
                json.dump(entry, f)
                f.write('\n')

        policy_manager_instance.update_action_preference = MagicMock()
        policy_manager_instance.save_policy = MagicMock()

        processed_count = policy_manager_instance.process_experience_log(temp_log_file)

        assert processed_count == 3
        assert policy_manager_instance.update_action_preference.call_count == 3

        # Check calls to update_action_preference
        state_key_A = policy_manager_instance._construct_state_key({"nlu_intent": "A"})
        state_key_B = policy_manager_instance._construct_state_key({"nlu_intent": "B"})

        calls = policy_manager_instance.update_action_preference.call_args_list
        assert calls[0][0] == (state_key_A, "Action1", 1.0) # args
        assert calls[0][1] == {"learning_rate": None}      # kwargs
        assert calls[1][0] == (state_key_B, "Action2", -0.5)
        assert calls[2][0] == (state_key_A, "Action1", 0.5)

        policy_manager_instance.save_policy.assert_called_once()

    def test_process_log_with_malformed_and_invalid_entries(self, policy_manager_instance: RLPolicyManager, temp_log_file: Path):
        log_content = (
            '{"state": {"nlu_intent": "C"}, "action_taken": "Action3", "calculated_reward": 0.8}\n' # Valid
            'this is not json\n' # Malformed
            '{"state": {"nlu_intent": "D"}, "action_taken": "Action4"}\n' # Missing reward
            '{"state": "not_a_dict", "action_taken": "Action5", "calculated_reward": 0.1}\n' # Invalid state type
            '{"state": {"nlu_intent": "E"}, "action_taken": null, "calculated_reward": 0.9}\n' # Invalid action type
        )
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)

        policy_manager_instance.update_action_preference = MagicMock()
        policy_manager_instance.save_policy = MagicMock()

        # Capture print warnings
        with patch('builtins.print') as mock_print:
            processed_count = policy_manager_instance.process_experience_log(temp_log_file)

        assert processed_count == 1 # Only the first entry is valid
        policy_manager_instance.update_action_preference.assert_called_once()
        state_key_C = policy_manager_instance._construct_state_key({"nlu_intent": "C"})
        policy_manager_instance.update_action_preference.assert_called_with(state_key_C, "Action3", 0.8, learning_rate=None)
        policy_manager_instance.save_policy.assert_called_once()

        # Check that warnings were printed for bad entries
        printed_warnings = [str(call_args[0]) for call_args in mock_print.call_args_list]
        assert any("Skipping malformed JSON log entry at line 2" in s for s in printed_warnings)
        assert any("Skipping log entry at line 3 due to missing state, action, or reward" in s for s in printed_warnings)
        assert any("Skipping log entry at line 4, 'state' is not a dictionary" in s for s in printed_warnings)
        assert any("Skipping log entry at line 5, 'action_taken' is not a string" in s for s in printed_warnings)

    def test_process_log_with_specific_learning_rate(self, policy_manager_instance: RLPolicyManager, temp_log_file: Path):
        log_entry_data = {"state": {"nlu_intent": "F"}, "action_taken": "ActionX", "calculated_reward": 1.0}
        with open(temp_log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry_data, f)
            f.write('\n')

        policy_manager_instance.update_action_preference = MagicMock()

        custom_lr = 0.5
        policy_manager_instance.process_experience_log(temp_log_file, learning_rate=custom_lr)

        state_key_F = policy_manager_instance._construct_state_key({"nlu_intent": "F"})
        policy_manager_instance.update_action_preference.assert_called_once_with(
            state_key_F, "ActionX", 1.0, learning_rate=custom_lr
        )

```
