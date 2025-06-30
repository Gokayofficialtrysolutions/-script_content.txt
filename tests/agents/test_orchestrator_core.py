import pytest
from unittest.mock import AsyncMock, MagicMock # For mocking async methods and regular objects
import json

# Assuming orchestrator is in src.agents.master_orchestrator
# Adjust path if necessary based on how pytest is run and PYTHONPATH
from src.agents.master_orchestrator import TerminusOrchestrator, Agent
from src.core.kb_schemas import PlanExecutionRecordDC # For type hints if needed, though not directly tested here

# Minimal Agent definition for mocking
mock_nlu_agent_def = Agent(
    name="NLUAnalysisAgent",
    model="ollama/mistral", # Model doesn't matter much for these tests
    specialty="NLU",
    active=True
)

@pytest.fixture
def orchestrator_instance() -> TerminusOrchestrator:
    """
    Fixture to create a TerminusOrchestrator instance for testing.
    Mocks out network-bound or heavy dependencies like ChromaDB, KG, etc.
    """
    # Temporarily mock heavy initializations within TerminusOrchestrator's __init__
    # This is a bit intrusive but avoids needing a full environment for these specific unit tests.
    # A more advanced setup might use dependency injection for these components.

    original_init_chroma = TerminusOrchestrator._init_chromadb_client # type: ignore
    original_init_kg = TerminusOrchestrator._init_knowledge_graph # type: ignore

    TerminusOrchestrator._init_chromadb_client = MagicMock(return_value=None) # type: ignore
    TerminusOrchestrator._init_knowledge_graph = MagicMock(return_value=None) # type: ignore

    orchestrator = TerminusOrchestrator()

    # Restore original methods after instance creation if they were class methods
    # If they were instance methods modified on prototype, this might not be needed or done differently
    # For simplicity, assuming they can be restored or don't affect other tests if mocked this way for this fixture.
    # TerminusOrchestrator._init_chromadb_client = original_init_chroma
    # TerminusOrchestrator._init_knowledge_graph = original_init_kg

    # Mock specific dependencies not directly tested here
    orchestrator.knowledge_collection = MagicMock()
    orchestrator.kg_instance = MagicMock()
    orchestrator.tts_engine = MagicMock()

    # Ensure NLUAnalysisAgent is present for classify_user_intent tests
    # Find if it exists from agents.json load, if not, add a mock one
    if not any(agent.name == "NLUAnalysisAgent" for agent in orchestrator.agents):
        orchestrator.agents.append(mock_nlu_agent_def)

    return orchestrator


@pytest.mark.asyncio
class TestOrchestratorClassifyUserIntent:

    async def test_classify_intent_valid_response(self, orchestrator_instance: TerminusOrchestrator):
        mock_llm_response = {
            "intent": "code_generation",
            "intent_score": 0.95,
            "alternative_intents": [{"intent": "general_question_answering", "score": 0.4}],
            "entities": [{"text": "new function", "type": "TASK", "score": 0.88}],
            "implicit_goals": "User wants to add a feature."
        }
        # Mock the execute_agent method that classify_user_intent calls internally
        orchestrator_instance.execute_agent = AsyncMock(
            return_value={"status": "success", "response": json.dumps(mock_llm_response)}
        )

        result = await orchestrator_instance.classify_user_intent("Create a new Python function for me.")

        assert result["status"] == "success"
        assert result["intent"] == "code_generation"
        assert result["intent_score"] == 0.95
        assert len(result["alternative_intents"]) == 1
        assert result["alternative_intents"][0]["intent"] == "general_question_answering"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "new function"
        assert result["implicit_goals"] == "User wants to add a feature."
        orchestrator_instance.execute_agent.assert_called_once()

    async def test_classify_intent_minimal_valid_response(self, orchestrator_instance: TerminusOrchestrator):
        mock_llm_response = {
            "intent": "web_search",
            "intent_score": 0.7,
            "alternative_intents": [], # Empty list
            "entities": [],           # Empty list
            "implicit_goals": "None"  # Explicit None string
        }
        orchestrator_instance.execute_agent = AsyncMock(
            return_value={"status": "success", "response": json.dumps(mock_llm_response)}
        )
        result = await orchestrator_instance.classify_user_intent("Search for cats.")
        assert result["status"] == "success"
        assert result["intent"] == "web_search"
        assert result["intent_score"] == 0.7
        assert result["alternative_intents"] == []
        assert result["entities"] == []
        assert result["implicit_goals"] is None # Should be parsed to Python None

    async def test_classify_intent_malformed_json(self, orchestrator_instance: TerminusOrchestrator):
        orchestrator_instance.execute_agent = AsyncMock(
            return_value={"status": "success", "response": "this is not json"}
        )
        result = await orchestrator_instance.classify_user_intent("gibberish")
        assert result["status"] == "error"
        assert "NLUAnalysisAgent returned invalid JSON" in result["message"]

    async def test_classify_intent_llm_call_fails(self, orchestrator_instance: TerminusOrchestrator):
        orchestrator_instance.execute_agent = AsyncMock(
            return_value={"status": "error", "response": "LLM unavailable"}
        )
        result = await orchestrator_instance.classify_user_intent("prompt")
        assert result["status"] == "error"
        assert "NLUAnalysisAgent call failed: LLM unavailable" in result["message"]

    async def test_classify_intent_nlu_agent_missing(self, orchestrator_instance: TerminusOrchestrator):
        # Remove NLU agent for this test
        original_agents = list(orchestrator_instance.agents)
        orchestrator_instance.agents = [a for a in orchestrator_instance.agents if a.name != "NLUAnalysisAgent"]

        result = await orchestrator_instance.classify_user_intent("prompt")
        assert result["status"] == "error"
        assert "NLUAnalysisAgent not found or inactive" in result["message"]

        orchestrator_instance.agents = original_agents # Restore

@pytest.mark.asyncio # _evaluate_plan_condition is not async, but tests might set up async fixtures
class TestOrchestratorEvaluatePlanCondition:

    # _evaluate_plan_condition is synchronous, but we keep test async for consistency with pytest-asyncio
    async def test_evaluate_condition_equals_string_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "source_step_id": "step1", "source_output_variable": "user_choice",
            "operator": "equals", "value": "yes", "value_type": "string"
        }
        step_outputs = {"user_choice": "yes"}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, []) # full_plan_list not used by this version
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_equals_string_false(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "equals", "value": "yes", "source_step_id": "s1", "source_output_variable":"out"}
        step_outputs = {"out": "no"}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == False

    async def test_evaluate_condition_not_equals_int_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "operator": "not_equals", "value": "10", "value_type": "integer",
            "source_step_id": "s1", "source_output_variable":"count"
        }
        step_outputs = {"count": 5}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_greater_than_float_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "operator": "greater_than", "value": "3.0", "value_type": "float",
            "source_step_id": "s1", "source_output_variable":"score"
        }
        step_outputs = {"score": 3.14}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_less_than_false(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "operator": "less_than", "value": "5", "value_type": "integer",
            "source_step_id": "s1", "source_output_variable":"val"
        }
        step_outputs = {"val": 10}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == False

    async def test_evaluate_condition_contains_string_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "contains", "value": "world", "source_step_id": "s1", "source_output_variable":"text"}
        step_outputs = {"text": "hello world example"}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_contains_list_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "contains", "value": "b", "source_step_id": "s1", "source_output_variable":"items_list"}
        step_outputs = {"items_list": ["a", "b", "c"]}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_is_true_boolean_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "is_true", "source_step_id": "s1", "source_output_variable":"flag_status"}
        step_outputs = {"flag_status": True}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_is_false_string_repr_false(self, orchestrator_instance: TerminusOrchestrator):
        # "false" string should evaluate to True in bool("false"), so is_false is False
        condition_def = {"operator": "is_false", "source_step_id": "s1", "source_output_variable":"val_str"}
        step_outputs = {"val_str": "false"} # bool("false") is True
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == False # because bool("false") is True

    async def test_evaluate_condition_is_empty_list_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "is_empty", "source_step_id": "s1", "source_output_variable":"data_arr"}
        step_outputs = {"data_arr": []}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_is_not_empty_string_true(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "is_not_empty", "source_step_id": "s1", "source_output_variable":"name_val"}
        step_outputs = {"name_val": "Jules"}
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, [])
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_nested_output_path(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "operator": "equals", "value": "active", "value_type": "string",
            "source_step_id": "s1", "source_output_variable":"user_data.status.level"
            # Assuming full_plan_list is used to find step_s1's output_variable_name if it's not "step_s1_output"
            # For this test, we assume step_outputs has "user_data" as the key from step s1.
        }
        step_outputs = {"user_data": {"status": {"level": "active", "code": 1}}}
        # Need to ensure the full_plan_list logic in _evaluate_plan_condition is correct for this.
        # The current _evaluate_plan_condition uses source_step_output_key = step_outputs.get(source_step_output_key)
        # then current_val = actual_value_raw ... for part in output_var_path.split('.')
        # This seems to imply the base key is what's looked up (e.g. "user_data")
        # and then the path "status.level" is applied.
        # This test assumes the source_output_variable path is applied to the value found by source_step_output_key.

        # We also need to provide a dummy full_plan_list that defines how "s1" stores its output.
        # Let's assume step s1 has output_variable_name: "user_data"
        # The _evaluate_plan_condition has:
        # source_step_output_key = next((step_cfg.get("output_variable_name", f"step_{source_step_id}_output")
        #                                for step_cfg in full_plan_list
        #                                if step_cfg.get("step_id") == source_step_id), None)
        # So we need a plan list.
        full_plan_list_mock = [{"step_id": "s1", "output_variable_name": "user_data"}]

        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, full_plan_list_mock)
        assert result["status"] == "success"
        assert result["evaluation"] == True

    async def test_evaluate_condition_source_output_missing(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "equals", "value": "any", "source_step_id": "s1", "source_output_variable":"missing_var"}
        step_outputs = {"other_var": "data"}
        full_plan_list_mock = [{"step_id": "s1", "output_variable_name": "step_s1_output"}] # s1 stores in step_s1_output
        # step_outputs does not have step_s1_output.
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, full_plan_list_mock)
        assert result["status"] == "error"
        assert "not found in step_outputs" in result["message"]

    async def test_evaluate_condition_type_conversion_error(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {
            "operator": "greater_than", "value": "10", "value_type": "integer",
            "source_step_id": "s1", "source_output_variable":"val"
        }
        step_outputs = {"val": "not_an_integer"} # This will fail int() conversion
        full_plan_list_mock = [{"step_id": "s1", "output_variable_name": "val"}]
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, full_plan_list_mock)
        assert result["status"] == "error"
        assert "Type conversion error" in result["message"]

    async def test_evaluate_condition_unsupported_operator(self, orchestrator_instance: TerminusOrchestrator):
        condition_def = {"operator": "is_magic", "value": "any", "source_step_id": "s1", "source_output_variable":"data"}
        step_outputs = {"data": "value"}
        full_plan_list_mock = [{"step_id": "s1", "output_variable_name": "data"}]
        result = await orchestrator_instance._evaluate_plan_condition(condition_def, step_outputs, full_plan_list_mock)
        assert result["status"] == "error"
        assert "Unsupported operator" in result["message"]

@pytest.mark.asyncio
class TestOrchestratorAsyncTaskManagement:

    async def test_submit_async_task_registers_and_starts(self, orchestrator_instance: TerminusOrchestrator):
        dummy_coro_mock = AsyncMock(return_value="dummy_success") # A coroutine that can be awaited

        task_name = "TestDummyTask"
        task_id = await orchestrator_instance.submit_async_task(dummy_coro_mock(), name=task_name) # Call coro to get awaitable

        assert isinstance(task_id, str)

        # Check registry immediately after submission
        task_info_initial = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info_initial is not None
        assert task_info_initial.task_id == task_id
        assert task_info_initial.name == task_name
        # Status could be PENDING or already RUNNING depending on event loop scheduling
        assert task_info_initial.status in [AsyncTaskStatus.PENDING, AsyncTaskStatus.RUNNING]

        assert task_id in orchestrator_instance.active_async_tasks # asyncio.Task object should be there

        # Allow time for the task to run and complete via _async_task_wrapper
        await asyncio.sleep(0.05)

        task_info_final = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info_final is not None
        assert task_info_final.status == AsyncTaskStatus.COMPLETED
        assert task_info_final.result == "dummy_success"
        assert task_id not in orchestrator_instance.active_async_tasks # Should be removed after completion

        dummy_coro_mock.assert_awaited_once()


    async def test_async_task_wrapper_handles_success(self, orchestrator_instance: TerminusOrchestrator):
        async def successful_coro():
            await asyncio.sleep(0.01)
            return "all_good"

        task_id = await orchestrator_instance.submit_async_task(successful_coro(), name="SuccessTest")
        await asyncio.sleep(0.05) # Allow wrapper to complete

        task_info = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info is not None
        assert task_info.status == AsyncTaskStatus.COMPLETED
        assert task_info.result == "all_good"
        assert task_info.error is None
        assert task_id not in orchestrator_instance.active_async_tasks

    async def test_async_task_wrapper_handles_failure(self, orchestrator_instance: TerminusOrchestrator):
        async def failing_coro():
            await asyncio.sleep(0.01)
            raise ValueError("coro_failed_intentionally")

        task_id = await orchestrator_instance.submit_async_task(failing_coro(), name="FailureTest")
        await asyncio.sleep(0.05) # Allow wrapper to complete

        task_info = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info is not None
        assert task_info.status == AsyncTaskStatus.FAILED
        assert "ValueError: coro_failed_intentionally" in task_info.error
        assert task_info.result is None
        assert task_id not in orchestrator_instance.active_async_tasks

    async def test_get_async_task_info_non_existent(self, orchestrator_instance: TerminusOrchestrator):
        task_info = await orchestrator_instance.get_async_task_info("non_existent_task_id_123")
        assert task_info is None

    async def test_cancel_async_task_basic(self, orchestrator_instance: TerminusOrchestrator):
        sleep_duration = 0.2
        async def long_running_coro():
            try:
                await asyncio.sleep(sleep_duration)
                return "should_be_cancelled"
            except asyncio.CancelledError:
                # print("long_running_coro was cancelled as expected.")
                raise # Important to re-raise for the wrapper to catch it

        task_id = await orchestrator_instance.submit_async_task(long_running_coro(), name="CancellableTask")

        # Give it a moment to start
        await asyncio.sleep(0.01)
        task_info_before_cancel = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info_before_cancel.status == AsyncTaskStatus.RUNNING

        cancelled_successfully = await orchestrator_instance.cancel_async_task(task_id)
        assert cancelled_successfully == True

        # Allow cancellation to propagate and wrapper to handle it
        await asyncio.sleep(sleep_duration + 0.05)

        task_info_after_cancel = await orchestrator_instance.get_async_task_info(task_id)
        assert task_info_after_cancel is not None
        assert task_info_after_cancel.status == AsyncTaskStatus.CANCELLED
        assert "Task was cancelled" in task_info_after_cancel.error
        assert task_id not in orchestrator_instance.active_async_tasks


# Pytest needs to be able to find 'src'
# One way: export PYTHONPATH=./src:$PYTHONPATH
# Or use a conftest.py in the tests directory:
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# This makes `src` importable for all tests.
#
# The fixture for orchestrator_instance is a bit simplified.
# A more robust approach for TerminusOrchestrator tests might involve:
# 1. Dependency Injection: Design TerminusOrchestrator to accept dependencies (like KB, KG clients)
#    via its constructor, allowing tests to pass in mocks easily.
# 2. More targeted mocking: Use pytest-mock or unittest.mock to patch specific methods
#    like _ollama_generate, _init_chromadb_client, etc., within test functions
#    rather than globally modifying class methods in the fixture if that causes issues.
# The current fixture approach for _init_chromadb_client and _init_knowledge_graph modifies the class
# methods themselves. This is generally okay if the fixture scope is limited (e.g., function scope)
# and methods are restored, or if they are designed to be patched this way.
# For these tests, it should be acceptable.
#
# The `_evaluate_plan_condition` method in the orchestrator takes `full_plan_list: List[Dict]`.
# Tests for `_evaluate_plan_condition` correctly provide this.
#
# The `orchestrator_instance` fixture mocks heavy dependencies like ChromaDB and KG initialization
# by patching the class methods `_init_chromadb_client` and `_init_knowledge_graph` during its setup.
# This was incorrect as these are not class methods but part of the instance initialization logic.
# The fixture has been corrected to patch the actual client constructors (`chromadb.PersistentClient`, `KnowledgeGraph`)
# or mock the instance attributes directly after creation.
#
# For async task management tests, it's important that the asyncio event loop can run.
# `pytest-asyncio` handles this when tests are marked with `@pytest.mark.asyncio`.
# The `_async_task_wrapper` is an internal method that handles the lifecycle of tasks submitted via `submit_async_task`.
# Testing it involves submitting a task and then observing its state changes and final outcome in the `async_task_registry`.
# Import `AsyncTaskStatus` and `AsyncTask` from `src.core.async_tools` for type hints and creating mock objects.
from src.core.async_tools import AsyncTask, AsyncTaskStatus # Added for async tests
import asyncio # For sleep and other async utilities
