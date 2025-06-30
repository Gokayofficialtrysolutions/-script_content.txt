"""
Unit tests for TerminusOrchestrator methods related to Knowledge Base (ChromaDB)
interactions, specifically store_knowledge and retrieve_knowledge.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import uuid # For generating unique IDs if needed in tests

from src.agents.master_orchestrator import TerminusOrchestrator, Agent
from src.core.kb_schemas import BaseKBSchema, PlanExecutionRecordDC, WebServiceScrapeResultDC # Import some schemas for testing
from src.core.event_system import SystemEvent # For asserting event payload

# Mock for the ChromaDB collection object
@pytest.fixture
def mock_chroma_collection():
    collection = MagicMock()
    collection.add = MagicMock()
    collection.query = MagicMock()
    collection.get = MagicMock() # Though not directly used by store/retrieve, good to have
    collection.update = MagicMock() # Same as above
    return collection

@pytest.fixture
def orchestrator_for_kb_tests(mock_chroma_collection: MagicMock) -> TerminusOrchestrator:
    """
    Fixture for testing orchestrator's ChromaDB KB interactions.
    Mocks ChromaDB client and other heavy dependencies.
    """
    # Patch the chromadb.PersistentClient constructor to return our mock_chroma_collection's parent client
    mock_persistent_client_instance = MagicMock()
    mock_persistent_client_instance.get_or_create_collection.return_value = mock_chroma_collection

    with patch('src.agents.master_orchestrator.chromadb.PersistentClient', return_value=mock_persistent_client_instance) as mock_chromadb_constructor, \
         patch('src.agents.master_orchestrator.TerminusOrchestrator._init_knowledge_graph', MagicMock(return_value=None)), \
         patch('src.agents.master_orchestrator.KnowledgeGraph', MagicMock()): # Ensure KG init is also mocked

        orchestrator = TerminusOrchestrator()

    # Replace other heavy components if necessary, though these tests focus on KB
    orchestrator.kg_instance = MagicMock()
    orchestrator.tts_engine = MagicMock()
    orchestrator.rl_logger = MagicMock()
    orchestrator.publish_event = AsyncMock() # Mock publish_event for store_knowledge tests

    # Ensure the mock collection is assigned
    orchestrator.knowledge_collection = mock_chroma_collection

    return orchestrator

@pytest.mark.asyncio
class TestOrchestratorStoreKnowledge:

    async def test_store_plain_text_content(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        content_id = "text_doc_001"
        text_content = "This is a simple test document."
        metadata = {"source": "test_plain_text", "custom_field": "value1"}

        result = await orchestrator_for_kb_tests.store_knowledge(
            content=text_content,
            metadata=metadata,
            content_id=content_id
        )

        assert result["status"] == "success"
        assert result["id"] == content_id

        mock_chroma_collection.add.assert_called_once()
        call_args = mock_chroma_collection.add.call_args[1] # kwargs
        assert call_args['ids'] == [content_id]
        assert call_args['documents'] == [text_content]
        assert call_args['metadatas'][0]["source"] == "test_plain_text"
        assert call_args['metadatas'][0]["custom_field"] == "value1"

        orchestrator_for_kb_tests.publish_event.assert_called_once()
        event_call_args = orchestrator_for_kb_tests.publish_event.call_args[1]
        assert event_call_args['event_type'] == "kb.content.added"
        assert event_call_args['payload']['kb_id'] == content_id
        assert event_call_args['payload']['metadata'] == metadata


    async def test_store_structured_content(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        scrape_id = f"scrape_{uuid.uuid4()}"
        structured_data = WebServiceScrapeResultDC(
            scrape_id=scrape_id, # This ID will be used if content_id is None
            url="http://example.com/scrape",
            title="Test Scrape",
            main_content_summary="Scraped summary.",
            source_agent_name="TestScraper"
        )
        metadata = {"source": "test_structured_content", "processed_by": "test_pipeline"}

        result = await orchestrator_for_kb_tests.store_knowledge(
            structured_content=structured_data,
            metadata=metadata
            # content_id will be taken from structured_data.scrape_id via final_id logic
        )

        final_id_used = result["id"] # This should be scrape_id if content_id was not passed
        assert result["status"] == "success"
        assert final_id_used == scrape_id

        mock_chroma_collection.add.assert_called_once()
        call_args = mock_chroma_collection.add.call_args[1]
        assert call_args['ids'] == [final_id_used]
        assert json.loads(call_args['documents'][0])["url"] == "http://example.com/scrape" # Check deserialized content
        assert call_args['metadatas'][0]["source"] == "test_structured_content"
        assert call_args['metadatas'][0]["kb_schema_type"] == "WebServiceScrapeResult" # Automatically added

        orchestrator_for_kb_tests.publish_event.assert_called_once()
        event_payload = orchestrator_for_kb_tests.publish_event.call_args[1]['payload']
        assert event_payload['kb_id'] == final_id_used
        assert event_payload['schema_type'] == "WebServiceScrapeResult"


    async def test_store_knowledge_generates_id_if_none_provided(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        result = await orchestrator_for_kb_tests.store_knowledge(content="Content without ID")
        assert result["status"] == "success"
        assert result["id"] is not None
        assert len(result["id"]) > 10 # UUIDs are long
        mock_chroma_collection.add.assert_called_once()
        assert mock_chroma_collection.add.call_args[1]['ids'] == [result["id"]]

    async def test_store_knowledge_kb_not_initialized(self, orchestrator_for_kb_tests: TerminusOrchestrator):
        orchestrator_for_kb_tests.knowledge_collection = None # Simulate KB init failure
        result = await orchestrator_for_kb_tests.store_knowledge(content="test")
        assert result["status"] == "error"
        assert "KB not initialized" in result["message"]

    async def test_store_knowledge_no_content_or_structured_content(self, orchestrator_for_kb_tests: TerminusOrchestrator):
        result = await orchestrator_for_kb_tests.store_knowledge()
        assert result["status"] == "error"
        assert "Either 'content' or 'structured_content' must be provided" in result["message"]

@pytest.mark.asyncio
class TestOrchestratorRetrieveKnowledge:

    async def test_retrieve_knowledge_success_no_schema(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        mock_query_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Text for doc1", "Text for doc2"]],
            "metadatas": [[{"source": "s1"}, {"source": "s2"}]],
            "distances": [[0.1, 0.2]]
        }
        mock_chroma_collection.query.return_value = mock_query_results

        results = await orchestrator_for_kb_tests.retrieve_knowledge("query text")

        assert results["status"] == "success"
        assert len(results["results"]) == 2
        assert results["results"][0]["id"] == "doc1"
        assert results["results"][0]["document_text"] == "Text for doc1"
        assert results["results"][0]["structured_document"] is None # No schema_type
        assert results["results"][0]["metadata"]["source"] == "s1"
        mock_chroma_collection.query.assert_called_once_with(
            query_texts=["query text"], n_results=5, where=None, include=["documents", "metadatas", "distances"]
        )

    async def test_retrieve_knowledge_with_schema_deserialization(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        plan_rec_data = PlanExecutionRecordDC(original_user_request="req", status="success", total_attempts=1, plan_json_executed_final_attempt="[]", final_summary_to_user="ok")
        plan_rec_json_str = plan_rec_data.to_json_string()

        mock_query_results = {
            "ids": [["plan1"]],
            "documents": [[plan_rec_json_str]],
            "metadatas": [[{"kb_schema_type": "PlanExecutionRecord", "source": "planner"}]],
            "distances": [[0.05]]
        }
        mock_chroma_collection.query.return_value = mock_query_results

        results = await orchestrator_for_kb_tests.retrieve_knowledge("find plans")

        assert results["status"] == "success"
        assert len(results["results"]) == 1
        retrieved_item = results["results"][0]
        assert retrieved_item["id"] == "plan1"
        assert isinstance(retrieved_item["structured_document"], PlanExecutionRecordDC)
        assert retrieved_item["structured_document"].original_user_request == "req"

    async def test_retrieve_knowledge_with_filter_and_n_results(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        mock_chroma_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]} # Empty result

        await orchestrator_for_kb_tests.retrieve_knowledge(
            "query with filter",
            n_results=3,
            filter_metadata={"source": "specific_source"}
        )

        mock_chroma_collection.query.assert_called_once_with(
            query_texts=["query with filter"],
            n_results=3,
            where={"source": "specific_source"},
            include=["documents", "metadatas", "distances"]
        )

    async def test_retrieve_knowledge_kb_not_initialized(self, orchestrator_for_kb_tests: TerminusOrchestrator):
        orchestrator_for_kb_tests.knowledge_collection = None # Simulate KB init failure
        results = await orchestrator_for_kb_tests.retrieve_knowledge("query")
        assert results["status"] == "error"
        assert "KB not initialized" in results["message"]

    async def test_retrieve_knowledge_empty_results_from_chroma(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        mock_chroma_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = await orchestrator_for_kb_tests.retrieve_knowledge("no results query")
        assert results["status"] == "success"
        assert len(results["results"]) == 0

    async def test_retrieve_knowledge_malformed_schema_data(self, orchestrator_for_kb_tests: TerminusOrchestrator, mock_chroma_collection: MagicMock):
        malformed_json_str = '{"original_user_request": "test", "status": "incomplete"}' # Missing fields for PlanExecutionRecordDC
        mock_query_results = {
            "ids": [["bad_plan_data"]],
            "documents": [[malformed_json_str]],
            "metadatas": [[{"kb_schema_type": "PlanExecutionRecord"}]],
            "distances": [[0.1]]
        }
        mock_chroma_collection.query.return_value = mock_query_results

        # Capture print output to check for warnings
        with patch('builtins.print') as mock_print:
            results = await orchestrator_for_kb_tests.retrieve_knowledge("fetch bad data")

        assert results["status"] == "success"
        assert len(results["results"]) == 1
        assert results["results"][0]["structured_document"] is None # Deserialization should fail gracefully

        # Check if warning was printed
        warning_found = False
        for call_args in mock_print.call_args_list:
            if "WARNING: Failed to deserialize" in call_args[0][0]:
                warning_found = True
                break
        assert warning_found, "Deserialization failure warning should have been printed."

```
