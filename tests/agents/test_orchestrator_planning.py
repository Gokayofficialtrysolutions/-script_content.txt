import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import datetime

from src.agents.master_orchestrator import TerminusOrchestrator, Agent
from src.core.kb_schemas import PlanExecutionRecordDC # For type hints if needed
from src.core.conversation_history import ConversationContextManager, ConversationTurn # For mocking
from src.core.rl_policy_manager import RLPolicyManager # Import for spec
from src.core.async_tools import AsyncTask, AsyncTaskStatus # Import for spec

# Minimal Agent definitions for mocking
mock_planner_agent_def = Agent(name="MasterPlanner", model="ollama/mock-planner", specialty="Planning", active=True)
mock_agent_A_def = Agent(name="AgentA", model="ollama/mock-agent-a", specialty="TaskA", active=True)
mock_agent_B_def = Agent(name="AgentB", model="ollama/mock-agent-b", specialty="TaskB", active=True)
mock_nlu_agent_def = Agent(name="NLUAnalysisAgent", model="ollama/mock-nlu", specialty="NLU", active=True)


@pytest.fixture
def mock_orchestrator() -> TerminusOrchestrator:
    """
    Fixture to create a TerminusOrchestrator instance for testing execute_master_plan.
    Heavy dependencies and external calls are mocked.
    """
    with patch('src.agents.master_orchestrator.TerminusOrchestrator._init_chromadb_client', MagicMock(return_value=None)), \
         patch('src.agents.master_orchestrator.TerminusOrchestrator._init_knowledge_graph', MagicMock(return_value=None)), \
         patch('src.agents.master_orchestrator.KnowledgeGraph', MagicMock()), \
         patch('src.agents.master_orchestrator.chromadb.PersistentClient', MagicMock()):

        orchestrator = TerminusOrchestrator()

    # Replace real components with mocks after __init__ has run
    orchestrator.knowledge_collection = MagicMock()
    orchestrator.kg_instance = MagicMock()
    orchestrator.kg_instance.get_source_nodes_related_to_target = AsyncMock(return_value=[]) # Default no KG results

    orchestrator.tts_engine = MagicMock()
    orchestrator.rl_logger = MagicMock() # Mock RL logger
    orchestrator.rl_policy_manager = MagicMock(spec=RLPolicyManager) # Add mock for RLPolicyManager

    # Ensure necessary agents are present, add if not (e.g. if agents.json is minimal/mocked)
    agent_names_present = [a.name for a in orchestrator.agents]
    if "MasterPlanner" not in agent_names_present: orchestrator.agents.append(mock_planner_agent_def)
    if "AgentA" not in agent_names_present: orchestrator.agents.append(mock_agent_A_def)
    if "AgentB" not in agent_names_present: orchestrator.agents.append(mock_agent_B_def)
    if "NLUAnalysisAgent" not in agent_names_present: orchestrator.agents.append(mock_nlu_agent_def)

    # Mock methods that would make external calls or have complex side effects not being tested here
    orchestrator._ollama_generate = AsyncMock() # For MasterPlanner's plan generation
    orchestrator.execute_agent = AsyncMock() # For individual agent step execution
    orchestrator.classify_user_intent = AsyncMock(return_value={ # Default NLU mock
        "status": "success", "intent": "test_intent", "intent_score": 0.9,
        "alternative_intents": [], "entities": [], "implicit_goals": "Test goal"
    })
    orchestrator.retrieve_knowledge = AsyncMock(return_value={"status": "success", "results": []}) # Default no KB results
    orchestrator._summarize_execution_for_user = AsyncMock(return_value="Plan executed (mock summary).")
    orchestrator._store_plan_execution_log_in_kb = AsyncMock(return_value="mock_kb_log_id_123")

    # Mock conversation context manager methods to avoid complex state
    orchestrator.conversation_context_manager = MagicMock(spec=ConversationContextManager)
    orchestrator.conversation_context_manager.extract_keywords_from_text = MagicMock(return_value=["keyword1", "keyword2"])
    orchestrator.conversation_context_manager.get_contextual_history = AsyncMock(
        return_value=MagicMock(total_token_estimate=100, selected_turns_count=2) # Mocked data object
    )
    orchestrator.conversation_context_manager.format_history_for_prompt = MagicMock(return_value="Mocked conversation history.")
    orchestrator.conversation_context_manager.update_full_history = MagicMock()
    orchestrator.conversation_context_manager.replace_turns_with_summary = MagicMock()
    orchestrator._orchestrate_conversation_summarization = AsyncMock()


    return orchestrator

@pytest.mark.asyncio
class TestExecuteMasterPlan:

    async def test_simple_sequential_plan_success(self, mock_orchestrator: TerminusOrchestrator):
        # 1. Mock MasterPlanner's response (the plan itself)
        plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Do task A", "output_variable_name": "outputA"},
            {"step_id": "2", "agent_name": "AgentB", "task_prompt": "Do task B using {{outputA}}", "dependencies": ["1"]}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        # 2. Mock individual agent executions
        async def mock_agent_exec_logic(agent: Agent, prompt: str, context=None):
            if agent.name == "AgentA":
                return {"status": "success", "response": "Result from Agent A"}
            elif agent.name == "AgentB":
                assert "Result from Agent A" in prompt # Check dependency substitution
                return {"status": "success", "response": "Result from Agent B"}
            return {"status": "error", "response": "Unknown agent in mock"}

        mock_orchestrator.execute_agent.side_effect = mock_agent_exec_logic

        # 3. Execute the plan
        user_prompt = "Run a simple two-step plan."
        final_results = await mock_orchestrator.execute_master_plan(user_prompt)

        # 4. Assertions
        assert len(final_results) == 2
        assert final_results[0]["status"] == "success"
        assert final_results[0]["agent_name"] == "AgentA"
        assert final_results[0]["response"] == "Result from Agent A"

        assert final_results[1]["status"] == "success"
        assert final_results[1]["agent_name"] == "AgentB"
        assert final_results[1]["response"] == "Result from Agent B"

        # Check that NLU, KB query, summarization, and logging were called
        mock_orchestrator.classify_user_intent.assert_called_once_with(user_prompt)
        mock_orchestrator.retrieve_knowledge.assert_called() # Called for general, plan logs, feedback
        mock_orchestrator._ollama_generate.assert_called_once() # For plan generation

        # execute_agent called twice (once for AgentA, once for AgentB)
        assert mock_orchestrator.execute_agent.call_count == 2

        mock_orchestrator._summarize_execution_for_user.assert_called_once()
        mock_orchestrator._store_plan_execution_log_in_kb.assert_called_once()

        # Check step_outputs (internal state, harder to check directly without access)
        # But we checked dependency substitution in AgentB's prompt.

    async def test_plan_with_dependencies(self, mock_orchestrator: TerminusOrchestrator):
        plan_json = [
            {"step_id": "s1", "agent_name": "AgentA", "task_prompt": "Task 1", "output_variable_name": "res1"},
            {"step_id": "s2", "agent_name": "AgentB", "task_prompt": "Task 2 (needs {{res1}})", "dependencies": ["s1"], "output_variable_name": "res2"},
            {"step_id": "s3", "agent_name": "AgentA", "task_prompt": "Task 3 (needs {{res2}} and {{res1}})", "dependencies": ["s1", "s2"]}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        agent_call_history = []
        async def mock_agent_exec_dependencies(agent: Agent, prompt: str, context=None):
            agent_call_history.append({"agent": agent.name, "prompt": prompt})
            if agent.name == "AgentA" and "Task 1" in prompt:
                return {"status": "success", "response": "OutputS1"}
            elif agent.name == "AgentB" and "Task 2" in prompt:
                assert "OutputS1" in prompt
                return {"status": "success", "response": "OutputS2"}
            elif agent.name == "AgentA" and "Task 3" in prompt:
                assert "OutputS1" in prompt
                assert "OutputS2" in prompt
                return {"status": "success", "response": "OutputS3"}
            return {"status": "error", "response": f"Unhandled mock for {agent.name} with prompt {prompt}"}

        mock_orchestrator.execute_agent.side_effect = mock_agent_exec_dependencies

        results = await mock_orchestrator.execute_master_plan("Dependent task plan")

        assert len(results) == 3
        assert results[0]["response"] == "OutputS1"
        assert results[1]["response"] == "OutputS2"
        assert results[2]["response"] == "OutputS3"

        # Ensure correct order of execution based on dependencies
        assert agent_call_history[0]["agent"] == "AgentA" and "Task 1" in agent_call_history[0]["prompt"]
        assert agent_call_history[1]["agent"] == "AgentB" and "Task 2" in agent_call_history[1]["prompt"]
        assert agent_call_history[2]["agent"] == "AgentA" and "Task 3" in agent_call_history[2]["prompt"]

    async def test_tool_suggestion_step_is_logged(self, mock_orchestrator: TerminusOrchestrator):
        plan_json = [
            {"step_id": "1", "agent_name": "SystemCapabilityManager", "task_prompt": "SUGGEST_NEW_TOOL",
             "suggested_tool_description": "Need a tool for advanced quantum entanglement analysis."}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        # Mock open to check log write
        # Using patch as a context manager for more localized mocking
        with patch("builtins.open", MagicMock()) as mock_open:
            results = await mock_orchestrator.execute_master_plan("Suggest a tool for me")

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["agent_name"] == "SystemCapabilityManager"
        assert results[0]["response"] == "Tool suggestion logged successfully."

        mock_open.assert_called_once_with(mock_orchestrator.logs_dir / "tool_suggestions.log", "a", encoding="utf-8")
        # Get the file handle mock from mock_open.return_value
        handle = mock_open.return_value.__enter__.return_value # For context manager behavior

        # Check if write was called on the file handle
        # The exact content check might be tricky if timestamp is involved and not mocked
        # For now, let's check that write was called.
        handle.write.assert_called_once()
        written_content = handle.write.call_args[0][0]
        assert "Tool Suggestion by MasterPlanner: Need a tool for advanced quantum entanglement analysis." in written_content

    # More tests to be added:
    # - test_plan_step_retry_logic_success_on_retry
    # - test_plan_step_retry_logic_failure_after_retries
    # - test_plan_failure_triggers_revision (mock ollama_generate to return a new plan on 2nd call)
    # - test_plan_revision_fails_if_max_revisions_reached
    # - test_agent_service_call_step_handling (mocking the service handler)
    # - test_plan_with_async_steps (mock execute_agent to return pending_async, then complete)

    # Note: Testing conditional and loop steps is more complex as it involves _handle_conditional_step
    # and _handle_loop_step which themselves have significant logic and might require more intricate
    # mocking of step_outputs and plan structures. These could be separate, focused tests.
    # The current `execute_master_plan` also has simplified/placeholder logic for loops and parallel groups.

    # The fixture `mock_orchestrator` is designed to simplify testing `execute_master_plan`
    # by isolating it from actual LLM calls, DB interactions, and complex conversation state.
    # It provides controllable returns for these dependencies.
    # Patching _init_chromadb_client and _init_knowledge_graph directly on the class
    # ensures that when TerminusOrchestrator() is called, these initializers are replaced by mocks.
    # This is a common technique for managing heavy dependencies during unit tests.
    # The patch context manager ensures the mocks are active only during the fixture's setup.
    # Direct assignment to orchestrator.knowledge_collection etc. after init further refines the mocks.

    # The conversation manager mocking is important because execute_master_plan interacts with it
    # for history and summarization.
    # We mock its methods to return predictable values and avoid its internal complexities.
    # `get_contextual_history` returns a mock object that mimics the structure of `ContextualHistoryData`.
    # This is important because the code accesses attributes like `total_token_estimate`.
    # The `spec=ConversationContextManager` in `MagicMock` helps ensure the mock conforms to the interface,
    # catching incorrect method calls if the interface changes.

    # The test for tool suggestion logging uses `patch("builtins.open", MagicMock())`.
    # This is a standard way to mock file I/O. `mock_open.return_value` gives the file handle mock.
    # `__enter__` is part of the context manager protocol for `with open(...)`.
    # `call_args[0][0]` gets the first positional argument of the first call to `write`.

    # Test for plan with dependencies checks the order of agent calls.
    # This implicitly tests if dependencies are resolved correctly before dispatching steps.

    # The tests here focus on the orchestration flow of execute_master_plan.
    # More detailed tests for specific helper methods like _execute_single_plan_step,
    # _handle_agent_service_call, _handle_conditional_step would be beneficial
    # but can be added iteratively.

    # The current version of _execute_single_plan_step is directly called by execute_master_plan.
    # Its logic (template substitution, retries) is implicitly tested when testing execute_master_plan
    # if the mocked execute_agent reflects these. However, direct unit tests for
    # _execute_single_plan_step with various template strings and retry scenarios would be more targeted.

    # For async steps: if execute_agent returns {"status": "pending_async", "task_id": ...},
    # execute_master_plan's main loop should poll get_async_task_info.
    # Testing this requires mocking get_async_task_info to first return PENDING/RUNNING, then COMPLETED/FAILED.
    # This will be a good candidate for a future test.

    # ... (existing comments remain, new tests will be added below) ...

    async def test_step_failure_no_retry_no_revision(self, mock_orchestrator: TerminusOrchestrator):
        plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Task A"},
            {"step_id": "2", "agent_name": "AgentB", "task_prompt": "Task B", "dependencies": ["1"]}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        async def mock_exec_failure(agent: Agent, prompt: str, context=None):
            if agent.name == "AgentA":
                return {"status": "error", "response": "Agent A failed catastrophically"}
            return {"status": "success", "response": "Should not be called"} # AgentB should not run

        mock_orchestrator.execute_agent.side_effect = mock_exec_failure
        mock_orchestrator.max_rev_attempts = 0 # Ensure no revision for this test

        results = await mock_orchestrator.execute_master_plan("Test step failure")

        assert len(results) == 1 # Only AgentA's result
        assert results[0]["status"] == "error"
        assert results[0]["agent_name"] == "AgentA"
        mock_orchestrator.execute_agent.assert_called_once_with(mock_agent_A_def, "Task A", None)

        # Check overall plan status (via rl_logger call)
        # rl_logger.log_experience(..., execution_status="failure", ...)
        # This requires capturing arguments to the mock.
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "failure"

    async def test_step_retry_succeeds(self, mock_orchestrator: TerminusOrchestrator):
        plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Retry Task",
             "max_retries": 1, "retry_delay_seconds": 0.01} # Quick retry
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        mock_orchestrator.execute_agent.side_effect = [
            {"status": "error", "response": "First attempt failed"}, # 1st call to AgentA
            {"status": "success", "response": "Succeeded on retry"}    # 2nd call to AgentA
        ]

        results = await mock_orchestrator.execute_master_plan("Test retry success")

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["response"] == "Succeeded on retry"
        assert mock_orchestrator.execute_agent.call_count == 2
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "success"

    async def test_step_retry_exhausted(self, mock_orchestrator: TerminusOrchestrator):
        plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Retry Task Exhaust",
             "max_retries": 2, "retry_delay_seconds": 0.01, "retry_on_statuses": ["error", "timeout"]}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        mock_orchestrator.execute_agent.side_effect = [
            {"status": "error", "response": "Attempt 1 fail"},
            {"status": "timeout", "response": "Attempt 2 timeout"},
            {"status": "error", "response": "Attempt 3 fail (final)"}
        ]

        results = await mock_orchestrator.execute_master_plan("Test retry exhaustion")

        assert len(results) == 1
        assert results[0]["status"] == "error" # Final status after retries
        assert results[0]["response"] == "Attempt 3 fail (final)"
        assert mock_orchestrator.execute_agent.call_count == 3 # Initial + 2 retries
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "failure"

    async def test_plan_revision_succeeds(self, mock_orchestrator: TerminusOrchestrator):
        initial_plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Initial Task - Will Fail"}
        ]
        revised_plan_json = [
            {"step_id": "1_rev", "agent_name": "AgentB", "task_prompt": "Revised Task - Will Succeed"}
        ]

        # MasterPlanner LLM (_ollama_generate) returns initial plan, then revised plan
        mock_orchestrator._ollama_generate.side_effect = [
            {"status": "success", "response": json.dumps(initial_plan_json)},
            {"status": "success", "response": json.dumps(revised_plan_json)}
        ]

        # Agent execution behavior
        mock_orchestrator.execute_agent.side_effect = [
            {"status": "error", "response": "AgentA initial failure"}, # For initial_plan_json's AgentA
            {"status": "success", "response": "AgentB revised success"} # For revised_plan_json's AgentB
        ]

        # execute_master_plan has max_rev_attempts = 1 by default
        results = await mock_orchestrator.execute_master_plan("Test plan revision success")

        assert len(results) == 1 # Results from the successful revised plan
        assert results[0]["status"] == "success"
        assert results[0]["agent_name"] == "AgentB" # From revised plan
        assert results[0]["response"] == "AgentB revised success"

        assert mock_orchestrator._ollama_generate.call_count == 2 # Initial plan + 1 revision
        assert mock_orchestrator.execute_agent.call_count == 2 # AgentA (fails) + AgentB (succeeds)
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "success"

    async def test_plan_revision_max_attempts_reached(self, mock_orchestrator: TerminusOrchestrator):
        initial_plan_json = [{"step_id": "1", "agent_name": "AgentA", "task_prompt": "Fail Plan 1"}]
        revised_plan_json = [{"step_id": "1r", "agent_name": "AgentA", "task_prompt": "Fail Plan 2 (Revised)"}]

        mock_orchestrator._ollama_generate.side_effect = [
            {"status": "success", "response": json.dumps(initial_plan_json)},
            {"status": "success", "response": json.dumps(revised_plan_json)}
        ]
        mock_orchestrator.execute_agent.return_value = {"status": "error", "response": "Persistent failure"}

        # Default max_rev_attempts = 1. So, initial + 1 revision. Both will fail.
        results = await mock_orchestrator.execute_master_plan("Test max revisions failed")

        assert len(results) == 1 # Contains the result of the last failed step of the last attempted plan
        assert results[0]["status"] == "error"
        assert mock_orchestrator._ollama_generate.call_count == 2 # Initial plan + 1 revision
        assert mock_orchestrator.execute_agent.call_count == 2 # One for each failing plan attempt
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "failure"
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['attempt_number'] == 1 # 0-indexed attempt in log for revisions

    # Placeholder for more tests as identified in comments

    async def test_single_async_step_completes_successfully(self, mock_orchestrator: TerminusOrchestrator):
        task_id_async = "async_task_001"
        plan_json = [
            {"step_id": "1", "agent_name": "AgentA", "task_prompt": "Async Task A", "output_variable_name": "outputA"}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        # AgentA's execute_agent call will initiate an async task
        mock_orchestrator.execute_agent.return_value = {
            "status": "pending_async", "task_id": task_id_async, "message": "Task for AgentA started."
        }

        # Mock get_async_task_info to simulate task progression
        mock_orchestrator.get_async_task_info = AsyncMock(side_effect=[
            MagicMock(spec=AsyncTask, task_id=task_id_async, status=AsyncTaskStatus.RUNNING, result=None, error=None), # First poll
            MagicMock(spec=AsyncTask, task_id=task_id_async, status=AsyncTaskStatus.RUNNING, result=None, error=None), # Second poll
            MagicMock(spec=AsyncTask, task_id=task_id_async, status=AsyncTaskStatus.COMPLETED,
                      result={"status": "success", "response": "Async Result A", "data": "Async Data A"}, # This is the result from the wrapped coroutine
                      error=None)
        ])

        results = await mock_orchestrator.execute_master_plan("Test single async success")

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["response"] == "Async Result A" # This comes from the 'response' within the AsyncTask.result
        # The 'data' field from AsyncTask.result is used for step_outputs if output_variable_name is present
        # To check step_outputs, we'd need to inspect mock_orchestrator.step_outputs (if it were exposed)
        # or infer from a dependent step. For a single step, checking the final result's response is okay.

        assert mock_orchestrator.execute_agent.call_count == 1 # Called once to initiate
        assert mock_orchestrator.get_async_task_info.call_count == 3 # Polled until completion

    async def test_single_async_step_fails(self, mock_orchestrator: TerminusOrchestrator):
        task_id_async = "async_task_002"
        plan_json = [{"step_id": "1", "agent_name": "AgentA", "task_prompt": "Async Task Fail"}]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        mock_orchestrator.execute_agent.return_value = {"status": "pending_async", "task_id": task_id_async}
        mock_orchestrator.get_async_task_info = AsyncMock(side_effect=[
            MagicMock(spec=AsyncTask, task_id=task_id_async, status=AsyncTaskStatus.RUNNING),
            MagicMock(spec=AsyncTask, task_id=task_id_async, status=AsyncTaskStatus.FAILED, error="Async task blew up")
        ])

        results = await mock_orchestrator.execute_master_plan("Test single async failure")

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["response"] == "Async task blew up"
        assert mock_orchestrator.rl_logger.log_experience.call_args[1]['execution_status'] == "failure"


    async def test_mixed_sync_async_sync_plan(self, mock_orchestrator: TerminusOrchestrator):
        async_task_id = "mixed_async_task"
        plan_json = [
            {"step_id": "s1", "agent_name": "AgentA", "task_prompt": "Sync Task 1", "output_variable_name": "out1"},
            {"step_id": "s2", "agent_name": "AgentB", "task_prompt": "Async Task 2 using {{out1}}", "dependencies": ["s1"], "output_variable_name": "out2_data"},
            {"step_id": "s3", "agent_name": "AgentA", "task_prompt": "Sync Task 3 using {{out2_data}}", "dependencies": ["s2"]}
        ]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}

        # execute_agent side effect
        async def mixed_exec_agent_side_effect(agent: Agent, prompt: str, context=None):
            if agent.name == "AgentA" and "Sync Task 1" in prompt:
                return {"status": "success", "response": "SyncOutput1"}
            elif agent.name == "AgentB" and "Async Task 2" in prompt:
                assert "SyncOutput1" in prompt # Check dependency
                return {"status": "pending_async", "task_id": async_task_id, "message": "AgentB async started"}
            elif agent.name == "AgentA" and "Sync Task 3" in prompt:
                assert "AsyncDataFromB" in prompt # Check dependency from async task's data
                return {"status": "success", "response": "SyncOutput3"}
            return {"status": "error", "response": "Unknown mock call in mixed_exec_agent"}
        mock_orchestrator.execute_agent.side_effect = mixed_exec_agent_side_effect

        # get_async_task_info side effect for AgentB's task
        mock_orchestrator.get_async_task_info = AsyncMock(side_effect=[
            MagicMock(spec=AsyncTask, task_id=async_task_id, status=AsyncTaskStatus.RUNNING), # Poll 1
            MagicMock(spec=AsyncTask, task_id=async_task_id, status=AsyncTaskStatus.COMPLETED,
                      result={"status": "success", "response": "Async Message From B", "data": "AsyncDataFromB"}, # This is the result of the service call / wrapped coroutine
                      error=None)  # Poll 2 - completes
        ])

        results = await mock_orchestrator.execute_master_plan("Mixed sync-async plan")

        assert len(results) == 3
        assert results[0]["response"] == "SyncOutput1"
        assert results[1]["status"] == "success"
        assert results[1]["response"] == "Async Message From B" # This is the message from the async step's result
        assert results[2]["response"] == "SyncOutput3"

        assert mock_orchestrator.execute_agent.call_count == 3 # s1, s2 (initiate), s3
        assert mock_orchestrator.get_async_task_info.call_count == 2 # Polled for s2

        # Check call order (more complex to assert directly with AsyncMock call_args for multiple calls)
        # But successful execution implies correct order due to dependencies.
        # We can check the prompts passed to execute_agent for sequence.
        agent_A_call1 = mock_orchestrator.execute_agent.call_args_list[0]
        agent_B_call = mock_orchestrator.execute_agent.call_args_list[1]
        agent_A_call2 = mock_orchestrator.execute_agent.call_args_list[2]

        assert "Sync Task 1" in agent_A_call1[0][1] # prompt is args[1] of the first arg tuple
        assert "Async Task 2 using SyncOutput1" in agent_B_call[0][1]
        assert "Sync Task 3 using AsyncDataFromB" in agent_A_call2[0][1]

    async def test_rl_dynamic_strategy_selection_specific_strategy(self, mock_orchestrator: TerminusOrchestrator):
        mock_orchestrator.rl_policy_manager._construct_state_key.return_value = "state_key_test_rl"
        mock_orchestrator.rl_policy_manager.get_best_action.return_value = "Strategy_FocusClarity"

        # Minimal plan to trigger the RL logging
        plan_json = [{"step_id": "1", "agent_name": "AgentA", "task_prompt": "Do something"}]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}
        mock_orchestrator.execute_agent.return_value = {"status": "success", "response": "Done"}

        await mock_orchestrator.execute_master_plan("Test RL strategy selection")

        expected_strategies = ["Strategy_Default", "Strategy_FocusClarity", "Strategy_PrioritizeBrevity"]
        mock_orchestrator.rl_policy_manager.get_best_action.assert_called_once_with("state_key_test_rl", expected_strategies)

        # Check that the selected strategy was logged
        logged_action = mock_orchestrator.rl_logger.log_experience.call_args[1]['action']
        assert logged_action == "Strategy_FocusClarity"
        logged_prompt_details = mock_orchestrator.rl_logger.log_experience.call_args[1]['master_planner_prompt_details']
        assert logged_prompt_details['strategy_used'] == "Strategy_FocusClarity"

    async def test_rl_dynamic_strategy_selection_get_best_action_returns_none(self, mock_orchestrator: TerminusOrchestrator):
        mock_orchestrator.rl_policy_manager._construct_state_key.return_value = "state_key_test_rl_none"
        mock_orchestrator.rl_policy_manager.get_best_action.return_value = None # Simulate no preferred action

        plan_json = [{"step_id": "1", "agent_name": "AgentA", "task_prompt": "Do something"}]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}
        mock_orchestrator.execute_agent.return_value = {"status": "success", "response": "Done"}

        await mock_orchestrator.execute_master_plan("Test RL strategy fallback to default")

        logged_action = mock_orchestrator.rl_logger.log_experience.call_args[1]['action']
        assert logged_action == "Strategy_Default" # Default strategy
        logged_prompt_details = mock_orchestrator.rl_logger.log_experience.call_args[1]['master_planner_prompt_details']
        assert logged_prompt_details['strategy_used'] == "Strategy_Default"

    async def test_rl_dynamic_strategy_selection_rl_manager_is_none(self, mock_orchestrator: TerminusOrchestrator):
        mock_orchestrator.rl_policy_manager = None # Simulate RL manager not being available

        plan_json = [{"step_id": "1", "agent_name": "AgentA", "task_prompt": "Do something"}]
        mock_orchestrator._ollama_generate.return_value = {"status": "success", "response": json.dumps(plan_json)}
        mock_orchestrator.execute_agent.return_value = {"status": "success", "response": "Done"}

        await mock_orchestrator.execute_master_plan("Test no RL manager")

        logged_action = mock_orchestrator.rl_logger.log_experience.call_args[1]['action']
        assert logged_action == "Strategy_Default" # Default strategy
        logged_prompt_details = mock_orchestrator.rl_logger.log_experience.call_args[1]['master_planner_prompt_details']
        assert logged_prompt_details['strategy_used'] == "Strategy_Default"
    pass
