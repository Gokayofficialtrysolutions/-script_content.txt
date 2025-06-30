import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.agents.master_orchestrator import TerminusOrchestrator, Agent, AgentServiceDefinition, AgentServiceParameter, AgentServiceReturn

# Minimal Agent definitions for mocking
mock_codemaster_agent_def = Agent(name="CodeMaster", model="ollama/mock-cm", specialty="Coding", active=True)
mock_docsummarizer_agent_def = Agent(name="DocSummarizer", model="ollama/mock-ds", specialty="Summarizing", active=True)

@pytest.fixture
def service_orchestrator() -> TerminusOrchestrator:
    """
    Fixture for testing orchestrator service handlers.
    Mocks out heavy dependencies and external calls.
    """
    with patch('src.agents.master_orchestrator.TerminusOrchestrator._init_chromadb_client', MagicMock(return_value=None)), \
         patch('src.agents.master_orchestrator.TerminusOrchestrator._init_knowledge_graph', MagicMock(return_value=None)), \
         patch('src.agents.master_orchestrator.KnowledgeGraph', MagicMock()), \
         patch('src.agents.master_orchestrator.chromadb.PersistentClient', MagicMock()):

        orchestrator = TerminusOrchestrator()

    orchestrator.knowledge_collection = MagicMock()
    orchestrator.kg_instance = MagicMock()
    orchestrator.tts_engine = MagicMock()

    # Ensure necessary agents are present
    agent_names_present = [a.name for a in orchestrator.agents]
    if "CodeMaster" not in agent_names_present: orchestrator.agents.append(mock_codemaster_agent_def)
    if "DocSummarizer" not in agent_names_present: orchestrator.agents.append(mock_docsummarizer_agent_def)

    orchestrator.execute_agent = AsyncMock() # Mock the common call point for these services
    return orchestrator

# Dummy service definitions to pass to handlers (normally loaded from agents.json)
codemaster_validate_service_def = AgentServiceDefinition(
    name="validate_code_syntax", description="Validates code.",
    parameters=[
        AgentServiceParameter(name="code_snippet", type="string", required=True, description="Code to validate."),
        AgentServiceParameter(name="language", type="string", required=False, description="Language.", default_value="python")
    ],
    returns=AgentServiceReturn(type="dict", description="Validation result.")
)

docsummarizer_summarize_service_def = AgentServiceDefinition(
    name="summarize_text", description="Summarizes text.",
    parameters=[
        AgentServiceParameter(name="text_to_summarize", type="string", required=True, description="Text to summarize.")
    ],
    returns=AgentServiceReturn(type="string", description="Summary.")
)


@pytest.mark.asyncio
class TestServiceCodeMasterValidateSyntax:

    async def test_validate_syntax_success(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "print('hello')", "language": "python"}
        mock_llm_response = {"is_valid": True, "errors": []}
        service_orchestrator.execute_agent.return_value = {
            "status": "success", "response": json.dumps(mock_llm_response)
        }

        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)

        assert result["status"] == "success"
        assert result["data"] == mock_llm_response
        assert "Syntax validation for python completed." in result["message"]
        service_orchestrator.execute_agent.assert_called_once()

    async def test_validate_syntax_failure_invalid_code(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "print 'hello'", "language": "python"} # Invalid Python 2 syntax
        mock_llm_response = {"is_valid": False, "errors": ["SyntaxError: Missing parentheses in call to 'print'"]}
        service_orchestrator.execute_agent.return_value = {
            "status": "success", "response": json.dumps(mock_llm_response)
        }

        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)

        assert result["status"] == "success" # Service itself succeeded, LLM provided valid error data
        assert result["data"]["is_valid"] == False
        assert len(result["data"]["errors"]) > 0

    async def test_validate_syntax_llm_returns_malformed_json(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "code", "language": "python"}
        service_orchestrator.execute_agent.return_value = {
            "status": "success", "response": "not json at all"
        }
        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "LLM_RESPONSE_MALFORMED"

    async def test_validate_syntax_llm_returns_incorrect_json_structure(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "code", "language": "python"}
        service_orchestrator.execute_agent.return_value = {
            "status": "success", "response": json.dumps({"valid": True}) # Missing 'is_valid' or 'errors'
        }
        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "LLM_RESPONSE_STRUCTURE_INVALID"

    async def test_validate_syntax_llm_call_fails(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "code", "language": "python"}
        service_orchestrator.execute_agent.return_value = {
            "status": "error", "response": "LLM agent exploded"
        }
        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "LLM_CALL_FAILED"
        assert "CodeMaster LLM call failed" in result["message"]

    async def test_validate_syntax_codemaster_agent_unavailable(self, service_orchestrator: TerminusOrchestrator):
        params = {"code_snippet": "code", "language": "python"}
        # Simulate CodeMaster agent being inactive or missing
        original_agents = list(service_orchestrator.agents)
        service_orchestrator.agents = [a for a in service_orchestrator.agents if a.name != "CodeMaster"]

        result = await service_orchestrator._service_codemaster_validate_syntax(params, codemaster_validate_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "AGENT_UNAVAILABLE"

        service_orchestrator.agents = original_agents # Restore

@pytest.mark.asyncio
class TestServiceDocSummarizerSummarizeText:

    async def test_summarize_text_success(self, service_orchestrator: TerminusOrchestrator):
        params = {"text_to_summarize": "This is a long text that needs to be summarized effectively."}
        expected_summary = "Short summary."
        service_orchestrator.execute_agent.return_value = {
            "status": "success", "response": expected_summary
        }

        result = await service_orchestrator._service_docsummarizer_summarize_text(params, docsummarizer_summarize_service_def)

        # This service handler propagates pending_async if execute_agent returns it.
        # For this test, we simulate execute_agent returning direct success (less likely for LLM).
        assert result["status"] == "success"
        assert result["data"] == expected_summary
        assert "Text summarized successfully (synchronous path)" in result["message"]
        service_orchestrator.execute_agent.assert_called_once()

    async def test_summarize_text_pending_async_propagation(self, service_orchestrator: TerminusOrchestrator):
        params = {"text_to_summarize": "Some text."}
        task_id = "summary_task_123"
        service_orchestrator.execute_agent.return_value = {
            "status": "pending_async", "task_id": task_id, "message": "Summarization task started."
        }

        result = await service_orchestrator._service_docsummarizer_summarize_text(params, docsummarizer_summarize_service_def)

        assert result["status"] == "pending_async"
        assert result["task_id"] == task_id
        assert "Summarization task initiated by DocSummarizer" in result["message"]

    async def test_summarize_text_empty_input(self, service_orchestrator: TerminusOrchestrator):
        params = {"text_to_summarize": "  "} # Empty or whitespace
        result = await service_orchestrator._service_docsummarizer_summarize_text(params, docsummarizer_summarize_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "MISSING_PARAMETER"
        service_orchestrator.execute_agent.assert_not_called()

    async def test_summarize_text_llm_call_fails_sync(self, service_orchestrator: TerminusOrchestrator):
        params = {"text_to_summarize": "Some text."}
        service_orchestrator.execute_agent.return_value = {
            "status": "error", "response": "DocSummarizer LLM exploded"
        }
        result = await service_orchestrator._service_docsummarizer_summarize_text(params, docsummarizer_summarize_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "LLM_CALL_FAILED_SYNC"
        assert "DocSummarizer LLM exploded" in result["message"]

    async def test_summarize_text_docsummarizer_agent_unavailable(self, service_orchestrator: TerminusOrchestrator):
        params = {"text_to_summarize": "Some text."}
        original_agents = list(service_orchestrator.agents)
        service_orchestrator.agents = [a for a in service_orchestrator.agents if a.name != "DocSummarizer"]

        result = await service_orchestrator._service_docsummarizer_summarize_text(params, docsummarizer_summarize_service_def)
        assert result["status"] == "error"
        assert result["error_code"] == "AGENT_UNAVAILABLE"

        service_orchestrator.agents = original_agents # Restore
```
