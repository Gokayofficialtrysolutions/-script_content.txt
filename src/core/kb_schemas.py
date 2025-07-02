import json
import uuid
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, TypeVar, Type

T = TypeVar('T')

class BaseKBSchema:
    """Base class for KB schema dataclasses to provide common (de)serialization."""
    def to_json_string(self) -> str:
        """Serializes the dataclass instance to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json_string(cls: Type[T], json_str: str) -> T:
        """Deserializes a JSON string to an instance of the dataclass."""
        data = json.loads(json_str)
        # For fields that are datetime, they need to be converted back from ISO strings
        # This generic approach might need specialization if nested dataclasses or complex types are used.
        # For now, assuming direct fields or simple nested dicts/lists.
        # Dataclasses don't auto-convert types from dicts, so this is a simplification.
        # A more robust solution would use a library like Pydantic or manually parse specific fields.
        # For this iteration, we rely on the caller to handle type consistency or the fields being basic types.
        # However, let's handle timestamps explicitly as they are common.

        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

        for key, value in data.items():
            if key in field_types:
                # Attempt to parse datetime strings if the type hint suggests datetime
                # This is a basic check; real datetime parsing is more complex for varied formats/timezones
                if str(field_types[key]).__contains__("datetime.datetime") and isinstance(value, str):
                    try:
                        data[key] = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        print(f"Warning: Could not parse timestamp for field {key}: {value}")
                        # Keep original string if parsing fails, or handle error differently
        try:
            return cls(**data)
        except TypeError as e:
            print(f"Error deserializing {cls.__name__} from data: {data}. Error: {e}")
            # This can happen if fields are missing or types are wrong and not handled above.
            # Consider raising the error or returning None/default instance.
            raise # Re-raise for now to make issues visible

@dataclass
class PlanExecutionRecordDC(BaseKBSchema):
    """
    Dataclass for storing structured information about a plan execution.
    """
    record_id: str = field(default_factory=lambda: f"planrec_{uuid.uuid4()}")
    chroma_db_id: Optional[str] = None # Populated by orchestrator after storage if needed
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    original_user_request: str
    primary_intent: Optional[str] = None
    nlu_analysis_raw: Optional[Dict[str, Any]] = None
    status: str # "success" | "failure"
    total_attempts: int
    plan_json_executed_final_attempt: str # JSON string of the plan
    final_summary_to_user: str
    step_results_summary: List[Dict[str, Any]] = field(default_factory=list) # e.g., {"step_id": "1", "status": "success", "response_preview": "..."}
    final_step_outputs: Dict[str, Any] = field(default_factory=dict) # Truncated outputs
    rl_interaction_id: Optional[str] = None

@dataclass
class CodeExplanationDC(BaseKBSchema):
    """
    Dataclass for storing structured explanations of code snippets.
    """
    explanation_id: str = field(default_factory=lambda: f"codeexp_{uuid.uuid4()}")
    chroma_db_id: Optional[str] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    language: str
    code_snippet: str # The actual code
    code_snippet_hash: Optional[str] = None # Optional: SHA256 hash of code_snippet for quick comparison
    explanation_text: str
    extracted_keywords: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    source_agent_name: Optional[str] = None

    def __post_init__(self):
        if self.code_snippet and not self.code_snippet_hash:
            import hashlib
            self.code_snippet_hash = hashlib.sha256(self.code_snippet.encode('utf-8')).hexdigest()

@dataclass
class WebServiceScrapeResultDC(BaseKBSchema):
    """
    Dataclass for storing structured results from web service scrapes (e.g., web pages).
    """
    scrape_id: str = field(default_factory=lambda: f"webscrape_{uuid.uuid4()}")
    chroma_db_id: Optional[str] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    url: str
    title: Optional[str] = None
    main_content_summary: str # The summarized text
    full_content_hash: Optional[str] = None # Hash of the full raw text, if available separately
    original_content_length: Optional[int] = None # Length of the full raw text
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{"text": "OpenAI", "type": "ORG"}]
    source_agent_name: Optional[str] = None # e.g., "WebCrawler"

@dataclass
class SimplifiedPlanStructureDC(BaseKBSchema):
    """
    Represents a simplified, structured summary of an executed plan,
    intended for storage in the Knowledge Graph and for learning purposes.
    """
    plan_id: str = field(default_factory=lambda: f"simplan_{uuid.uuid4()}")
    # Typically, this would be linked to a PlanExecutionRecordDC.record_id
    # If this simplified plan is stored as its own primary KB item (e.g. in Chroma),
    # it might have its own chroma_db_id.
    # For now, assuming it's primarily a KG node document linked to a PlanExecutionRecord.
    original_request_preview: Optional[str] = None
    primary_intent: Optional[str] = None
    status: str = "unknown"  # "success", "failure"
    num_steps: int = 0
    agent_sequence: List[str] = field(default_factory=list) # List of agent names in order of execution
    key_abstractions_or_entities: List[str] = field(default_factory=list) # Key concepts or entities involved in the plan
    feedback_rating_if_any: Optional[str] = None # "positive", "negative", "neutral", "none"
    # timestamp_utc is inherited from BaseKBSchema and should be set on creation

    # No separate from_json_string needed if BaseKBSchema's version is sufficient
    # and all fields are simple types or handled by BaseKBSchema.
    # If custom logic is needed (e.g. for nested dataclasses not handled by base), override it.
    # For now, assuming base is okay.

@dataclass
class GenericDocumentDC(BaseKBSchema):
    """
    Dataclass for storing generic document content and its summary.
    Useful for user-uploaded documents or other text not fitting specific schemas.
    """
    record_id: str = field(default_factory=lambda: f"gendoc_{uuid.uuid4()}")
    chroma_db_id: Optional[str] = None # Populated by orchestrator after storage if needed
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    source_identifier: str # e.g., filename, URL, or user-provided description
    original_content: str # The full original text content
    summary_content: str  # The summarized version of the original_content
    original_content_hash: Optional[str] = None # Optional: SHA256 hash of original_content
    processing_notes: Optional[str] = None # Any notes from the agent that processed/summarized this
    source_agent_name: Optional[str] = None # e.g., "DocProcessor" or "UserUploadHandler"

    def __post_init__(self):
        if self.original_content and not self.original_content_hash:
            import hashlib
            self.original_content_hash = hashlib.sha256(self.original_content.encode('utf-8')).hexdigest()

@dataclass
class UserObjectiveDC(BaseKBSchema):
    """
    Dataclass for storing a user's long-term objective.
    """
    objective_id: str = field(default_factory=lambda: f"userobj_{uuid.uuid4()}")
    user_identifier: str  # Associates with a user, session, or project.
    description: str      # Textual description of the objective.
    status: str           # e.g., "active", "on_hold", "completed", "archived".
    priority: int = 3     # Default priority (e.g., 1-High, 5-Low).
    created_timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    last_updated_timestamp_utc: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    completion_timestamp_utc: Optional[str] = None
    related_project_id: Optional[str] = None
    key_details: Dict[str, Any] = field(default_factory=dict) # For additional structured info.
    notes: Optional[str] = None

    def __post_init__(self):
        # Ensure last_updated is set if not explicitly passed (e.g. on creation from_json_string)
        if not hasattr(self, 'last_updated_timestamp_utc') or not self.last_updated_timestamp_utc:
            self.last_updated_timestamp_utc = datetime.datetime.utcnow().isoformat()
        if not hasattr(self, 'created_timestamp_utc') or not self.created_timestamp_utc:
             self.created_timestamp_utc = datetime.datetime.utcnow().isoformat()


    def to_dict_for_prompt(self) -> Dict[str, Any]:
        """Returns a simplified dict suitable for including in an LLM prompt context."""
        prompt_dict = {
            "id": self.objective_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "last_updated": self.last_updated_timestamp_utc
        }
        if self.key_details:
            prompt_dict["key_details"] = self.key_details
        return prompt_dict

    def mark_updated(self):
        self.last_updated_timestamp_utc = datetime.datetime.utcnow().isoformat()

if __name__ == '__main__':
    print("--- Testing PlanExecutionRecordDC ---")
    plan_rec = PlanExecutionRecordDC(
        original_user_request="Test request",
        primary_intent="test_intent",
        status="success",
        total_attempts=1,
        plan_json_executed_final_attempt='[{"step_id": "1", "action": "do_something"}]',
        final_summary_to_user="It worked!",
        step_results_summary=[{"step_id": "1", "status": "success", "response_preview": "Done."}],
        final_step_outputs={"output1": "result1"}
    )
    plan_json_str = plan_rec.to_json_string()
    print(f"Serialized Plan Record:\n{plan_json_str}")
    plan_rec_restored = PlanExecutionRecordDC.from_json_string(plan_json_str)
    assert plan_rec_restored.original_user_request == plan_rec.original_user_request
    assert plan_rec_restored.status == "success"
    print("PlanExecutionRecordDC tests passed.\n")

    print("--- Testing CodeExplanationDC ---")
    code_exp = CodeExplanationDC(
        language="python",
        code_snippet="def hello():\n  print('world')",
        explanation_text="This function prints 'world'.",
        extracted_keywords=["python", "print", "function"],
        source_agent_name="CodeMaster"
    )
    code_exp_json_str = code_exp.to_json_string()
    print(f"Serialized Code Explanation:\n{code_exp_json_str}")
    code_exp_restored = CodeExplanationDC.from_json_string(code_exp_json_str)
    assert code_exp_restored.language == "python"
    assert code_exp_restored.code_snippet_hash is not None
    print("CodeExplanationDC tests passed.\n")

    print("--- Testing WebServiceScrapeResultDC ---")
    web_scrape = WebServiceScrapeResultDC(
        url="https://example.com",
        title="Example Domain",
        main_content_summary="This domain is for use in illustrative examples in documents.",
        extracted_entities=[{"text": "example", "type": "CONCEPT"}],
        source_agent_name="WebCrawler"
    )
    web_scrape_json_str = web_scrape.to_json_string()
    print(f"Serialized Web Scrape Result:\n{web_scrape_json_str}")
    web_scrape_restored = WebServiceScrapeResultDC.from_json_string(web_scrape_json_str)
    assert web_scrape_restored.url == "https://example.com"
    assert len(web_scrape_restored.extracted_entities) == 1
    print("WebServiceScrapeResultDC tests passed.\n")

    print("All kb_schemas.py basic tests passed.")
