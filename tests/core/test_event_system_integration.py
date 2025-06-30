"""
Unit tests for the core event system (publish, subscribe, dispatch loop)
and key event handlers within the TerminusOrchestrator, primarily
focusing on the integration and behavior of _event_handler_kb_content_added.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio # For event loop and task management in tests

from src.agents.master_orchestrator import TerminusOrchestrator, Agent
from src.core.event_system import SystemEvent
from src.core.kb_schemas import PlanExecutionRecordDC # For schema type testing

# Minimal Agent definition for mocking
mock_content_analysis_agent_def = Agent(name="ContentAnalysisAgent", model="ollama/mock-ca", specialty="Content Analysis", active=True)

@pytest.fixture
async def orchestrator_for_event_tests() -> TerminusOrchestrator:
    """
    Fixture for testing orchestrator's event system and core handlers.
    Mocks ChromaDB, KG, and other heavy dependencies.
    Ensures event dispatcher loop is started and can be gracefully stopped.
    """
    mock_persistent_client_instance = MagicMock()
    mock_chroma_collection = MagicMock()
    mock_persistent_client_instance.get_or_create_collection.return_value = mock_chroma_collection

    with patch('src.agents.master_orchestrator.chromadb.PersistentClient', return_value=mock_persistent_client_instance), \
         patch('src.agents.master_orchestrator.KnowledgeGraph') as MockKG: # Patch KG class

        # Mock the KG instance that would be created
        mock_kg_instance = MagicMock()
        mock_kg_instance.add_node = MagicMock()
        mock_kg_instance.add_edge = MagicMock()
        MockKG.return_value = mock_kg_instance # Ensure constructor returns our mock

        orchestrator = TerminusOrchestrator()

    orchestrator.knowledge_collection = mock_chroma_collection # Used by _event_handler_kb_content_added
    orchestrator.kg_instance = mock_kg_instance # Used by _event_handler_kb_content_added
    orchestrator.tts_engine = MagicMock()
    orchestrator.rl_logger = MagicMock()
    orchestrator.execute_agent = AsyncMock() # For ContentAnalysisAgent call
    orchestrator._update_kb_item_metadata = AsyncMock(return_value={"status": "success"}) # For _event_handler_kb_content_added

    # Ensure ContentAnalysisAgent is present
    if not any(agent.name == "ContentAnalysisAgent" for agent in orchestrator.agents):
        orchestrator.agents.append(mock_content_analysis_agent_def)

    # Allow event dispatcher to run a bit then stop it for cleanup
    # The dispatcher loop starts in TerminusOrchestrator.__init__
    # We need a way to signal it to stop or cancel it.
    yield orchestrator

    # Cleanup: Cancel the event dispatcher task
    if orchestrator._event_dispatcher_task and not orchestrator._event_dispatcher_task.done():
        orchestrator._event_dispatcher_task.cancel()
        try:
            await orchestrator._event_dispatcher_task # Wait for cancellation to complete
        except asyncio.CancelledError:
            pass # Expected

    # Close KG if it had a real close method (mocked here)
    if orchestrator.kg_instance and hasattr(orchestrator.kg_instance, 'close'):
         orchestrator.kg_instance.close()


@pytest.mark.asyncio
class TestEventSystem:

    async def test_publish_subscribe_dispatch(self, orchestrator_for_event_tests: TerminusOrchestrator):
        orchestrator = orchestrator_for_event_tests
        mock_handler = AsyncMock()
        event_type = "test.event.occurred"
        payload = {"data": "test_payload", "value": 123}

        orchestrator.subscribe_to_event(event_type, mock_handler)

        event_id = await orchestrator.publish_event(
            event_type=event_type,
            source_component="TestComponent",
            payload=payload
        )

        # Give the event dispatcher loop a chance to process the event
        await asyncio.sleep(0.01) # Small delay for the event to be processed

        mock_handler.assert_called_once()
        called_event: SystemEvent = mock_handler.call_args[0][0]

        assert isinstance(called_event, SystemEvent)
        assert called_event.event_id == event_id
        assert called_event.event_type == event_type
        assert called_event.source_component == "TestComponent"
        assert called_event.payload == payload

    async def test_multiple_handlers_for_one_event(self, orchestrator_for_event_tests: TerminusOrchestrator):
        orchestrator = orchestrator_for_event_tests
        mock_handler1 = AsyncMock()
        mock_handler2 = AsyncMock()
        event_type = "multi.handler.event"

        orchestrator.subscribe_to_event(event_type, mock_handler1)
        orchestrator.subscribe_to_event(event_type, mock_handler2)

        await orchestrator.publish_event(event_type, "TestSource", {"info": "multi"})
        await asyncio.sleep(0.01)

        mock_handler1.assert_called_once()
        mock_handler2.assert_called_once()

    async def test_no_handler_for_event(self, orchestrator_for_event_tests: TerminusOrchestrator):
        orchestrator = orchestrator_for_event_tests
        # No print/log capture here, but we ensure no error occurs and queue gets processed
        await orchestrator.publish_event("unhandled.event", "Test", {})
        await asyncio.sleep(0.01) # Allow dispatcher to process
        # No assertion needed other than it didn't crash

@pytest.mark.asyncio
class TestEventHandlerKbContentAdded:

    @pytest.fixture
    def sample_kb_added_event(self) -> SystemEvent:
        return SystemEvent(
            event_type="kb.content.added",
            source_component="TestKBStorage",
            payload={
                "kb_id": "kb_doc_123",
                "metadata": {"source": "unit_test_source", "kb_schema_type": "GenericContent"},
                "content_preview": "This is some generic content for testing analysis."
            }
        )

    async def test_handler_processes_generic_content(self, orchestrator_for_event_tests: TerminusOrchestrator, sample_kb_added_event: SystemEvent):
        orchestrator = orchestrator_for_event_tests
        kb_id = sample_kb_added_event.payload["kb_id"]

        # Mock KB collection get
        mock_doc_string = sample_kb_added_event.payload["content_preview"]
        mock_db_metadata = sample_kb_added_event.payload["metadata"]
        orchestrator.knowledge_collection.get.return_value = {
            "ids": [[kb_id]], "documents": [[mock_doc_string]], "metadatas": [[mock_db_metadata]]
        }

        # Mock ContentAnalysisAgent response
        analysis_response = {"keywords": "test, analysis, content", "topics": "Testing, NLP"}
        orchestrator.execute_agent.return_value = { # Simulating direct LLM call success for ContentAnalysisAgent
            "status": "success", "response": json.dumps(analysis_response)
        }

        await orchestrator._event_handler_kb_content_added(sample_kb_added_event)

        orchestrator.knowledge_collection.get.assert_called_once_with(ids=[kb_id], include=["documents", "metadatas"])
        orchestrator.execute_agent.assert_called_once() # Called for ContentAnalysisAgent

        # Check metadata update call
        orchestrator._update_kb_item_metadata.assert_called_once()
        update_call_args = orchestrator._update_kb_item_metadata.call_args[0]
        assert update_call_args[0] == kb_id
        assert update_call_args[1]["extracted_keywords"] == "test, analysis, content"
        assert update_call_args[1]["extracted_topics"] == "Testing, NLP"

        # Check KG calls
        orchestrator.kg_instance.add_node.assert_any_call(node_id=kb_id, node_type="GenericContent", content_preview=mock_doc_string[:100])
        orchestrator.kg_instance.add_node.assert_any_call(node_id="keyword_test", node_type="Keyword", content_preview="test")
        orchestrator.kg_instance.add_edge.assert_any_call(source_node_id=kb_id, target_node_id="keyword_test", relationship_type="HAS_KEYWORD", ensure_nodes=False)
        orchestrator.kg_instance.add_node.assert_any_call(node_id="topic_testing", node_type="Topic", content_preview="testing")
        orchestrator.kg_instance.add_edge.assert_any_call(source_node_id=kb_id, target_node_id="topic_testing", relationship_type="HAS_TOPIC", ensure_nodes=False)


    async def test_handler_skips_feedback_report_schema(self, orchestrator_for_event_tests: TerminusOrchestrator, sample_kb_added_event: SystemEvent):
        orchestrator = orchestrator_for_event_tests
        # Modify event to be a feedback report
        sample_kb_added_event.payload["metadata"]["kb_schema_type"] = "FeedbackReport"
        sample_kb_added_event.payload["metadata"]["source"] = "feedback_analysis_report"

        await orchestrator._event_handler_kb_content_added(sample_kb_added_event)

        orchestrator.knowledge_collection.get.assert_not_called()
        orchestrator.execute_agent.assert_not_called()
        orchestrator._update_kb_item_metadata.assert_not_called()
        orchestrator.kg_instance.add_node.assert_not_called()


    async def test_handler_gracefully_handles_missing_kb_item(self, orchestrator_for_event_tests: TerminusOrchestrator, sample_kb_added_event: SystemEvent):
        orchestrator = orchestrator_for_event_tests
        orchestrator.knowledge_collection.get.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]} # Empty result

        await orchestrator._event_handler_kb_content_added(sample_kb_added_event)

        orchestrator.execute_agent.assert_not_called() # Should not proceed to analysis

    async def test_handler_handles_analysis_agent_failure(self, orchestrator_for_event_tests: TerminusOrchestrator, sample_kb_added_event: SystemEvent):
        orchestrator = orchestrator_for_event_tests
        kb_id = sample_kb_added_event.payload["kb_id"]
        orchestrator.knowledge_collection.get.return_value = {
            "ids": [[kb_id]], "documents": [["Some text"]], "metadatas": [[sample_kb_added_event.payload["metadata"]]]
        }
        orchestrator.execute_agent.return_value = {"status": "error", "response": "Analysis failed"}

        await orchestrator._event_handler_kb_content_added(sample_kb_added_event)

        orchestrator._update_kb_item_metadata.assert_not_called() # Should not update if analysis fails
        # KG calls for keywords/topics should also not happen
        # Check that no "HAS_KEYWORD" or "HAS_TOPIC" edges were attempted for kb_id
        add_edge_calls = orchestrator.kg_instance.add_edge.call_args_list
        for call in add_edge_calls:
            if call[1].get('source_node_id') == kb_id: # kwargs is at index 1
                 assert call[1].get('relationship_type') not in ["HAS_KEYWORD", "HAS_TOPIC"]

    # Test specific schema parsing for text_for_analysis (e.g., PlanExecutionRecord)
    async def test_handler_parses_plan_execution_record_for_analysis(self, orchestrator_for_event_tests: TerminusOrchestrator, sample_kb_added_event: SystemEvent):
        orchestrator = orchestrator_for_event_tests
        kb_id = "plan_rec_test_id"
        plan_data = PlanExecutionRecordDC(
            original_user_request="Test user request for plan.",
            status="success", total_attempts=1,
            plan_json_executed_final_attempt="[]",
            final_summary_to_user="Plan was successful."
        )
        plan_json_str = plan_data.to_json_string()

        sample_kb_added_event.payload["kb_id"] = kb_id
        sample_kb_added_event.payload["metadata"]["kb_schema_type"] = "PlanExecutionRecord"

        orchestrator.knowledge_collection.get.return_value = {
            "ids": [[kb_id]], "documents": [[plan_json_str]], "metadatas": [[sample_kb_added_event.payload["metadata"]]]
        }
        orchestrator.execute_agent.return_value = {
            "status": "success", "response": json.dumps({"keywords": "plan, success", "topics": "Planning"})
        }

        await orchestrator._event_handler_kb_content_added(sample_kb_added_event)

        orchestrator.execute_agent.assert_called_once()
        analysis_prompt = orchestrator.execute_agent.call_args[0][1] # Second arg to execute_agent is the prompt
        assert "User Request: Test user request for plan." in analysis_prompt
        assert "Outcome Summary: Plan was successful." in analysis_prompt
        orchestrator._update_kb_item_metadata.assert_called_once()
```
