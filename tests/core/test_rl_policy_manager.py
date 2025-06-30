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

@pytest.mark.asyncio
class TestRLPolicyManagerEventHandling:
    """
    Tests the event subscription and handling capabilities of RLPolicyManager,
    specifically its reaction to 'rl.experience.logged' events.
    """
    def test_init_subscribes_to_event_if_bus_and_path_provided(self, temp_policy_file: Path, temp_log_file: Path):
        mock_event_bus = MagicMock()
        mock_event_bus.subscribe_to_event = MagicMock() # Ensure it has the method

        # Instantiate RLPolicyManager with the mock event bus and a log file path
        manager = RLPolicyManager(
            policy_storage_path=temp_policy_file,
            event_bus=mock_event_bus,
            experience_log_file_path=temp_log_file
        )

        mock_event_bus.subscribe_to_event.assert_called_once()
        args, _ = mock_event_bus.subscribe_to_event.call_args
        assert args[0] == "rl.experience.logged"
        # Check that the handler is the correct bound method. This is a bit tricky.
        # We can check if it's a callable and its __self__ is the manager instance.
        assert callable(args[1])
        assert getattr(args[1], '__self__', None) == manager
        # More specific: assert args[1] == manager._handle_experience_logged_event
        # but this requires knowing the exact method name and it might be bound differently.
        # A common way is to check its name if it's not a lambda or partial.
        assert args[1].__name__ == 'typed_handler' # It's the wrapper
        # To check the wrapped method, it's more involved. For unit test, checking it's a bound method of manager is good.


    def test_init_does_not_subscribe_if_event_bus_missing(self, temp_policy_file: Path, temp_log_file: Path):
        manager = RLPolicyManager(
            policy_storage_path=temp_policy_file,
            event_bus=None, # No event bus
            experience_log_file_path=temp_log_file
        )
        # If event_bus was a mock, we'd assert not_called. Here, we just ensure no error.
        assert manager.event_bus is None

    def test_init_does_not_subscribe_if_log_path_missing(self, temp_policy_file: Path):
        mock_event_bus = MagicMock()
        mock_event_bus.subscribe_to_event = MagicMock()

        manager = RLPolicyManager(
            policy_storage_path=temp_policy_file,
            event_bus=mock_event_bus,
            experience_log_file_path=None # No log path
        )
        mock_event_bus.subscribe_to_event.assert_not_called()
        assert manager.experience_log_file_path is None


    async def test_handle_experience_logged_event_calls_process_log(self, temp_policy_file: Path, temp_log_file: Path):
        mock_event_bus = MagicMock() # Not used for subscription check here, direct handler call
        manager = RLPolicyManager(
            policy_storage_path=temp_policy_file,
            event_bus=None, # Don't subscribe for this direct handler test
            experience_log_file_path=temp_log_file # Must be provided for handler to work
        )

        # Mock the method that would be called by the handler's task
        manager.process_experience_log = MagicMock(return_value=1)

        # Mock asyncio.create_task to immediately execute the coroutine (or just run the inner part)
        # For simplicity in this unit test, we'll patch process_experience_log and assume create_task works.
        # The goal is to test that _handle_experience_logged_event *tries* to call process_experience_log.

        mock_event = MagicMock(spec=SystemEvent)
        mock_event.event_type = "rl.experience.logged"
        mock_event.event_id = "test_event_id_123"
        mock_event.payload = {"log_file_path": str(temp_log_file)} # Example payload

        # Patch asyncio.create_task to verify it's called with a coroutine
        # that eventually calls process_experience_log
        with patch('asyncio.create_task') as mock_create_task:
            # Define a side effect for create_task that immediately executes the passed coroutine
            async def immediate_execute_task(coro, *args, **kwargs):
                await coro # Execute the coroutine passed to create_task
                return MagicMock() # Return a mock task object

            mock_create_task.side_effect = immediate_execute_task

            await manager._handle_experience_logged_event(mock_event)

        mock_create_task.assert_called_once() # Ensure it tried to create a task
        manager.process_experience_log.assert_called_once_with(temp_log_file)


    async def test_handle_experience_logged_event_no_log_path(self, temp_policy_file: Path):
        manager = RLPolicyManager(
            policy_storage_path=temp_policy_file,
            experience_log_file_path=None # Crucial: no log path configured
        )
        manager.process_experience_log = MagicMock()

        mock_event = MagicMock(spec=SystemEvent)

        with patch('builtins.print') as mock_print: # Capture print warnings
            await manager._handle_experience_logged_event(mock_event)

        manager.process_experience_log.assert_not_called()
        assert any("Warning - _handle_experience_logged_event called but no experience_log_file_path configured" in str(c) for c in mock_print.call_args_list)
```
```
