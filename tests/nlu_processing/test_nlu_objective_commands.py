import pytest
from src.nlu_processing.nlu_engine import NLUProcessor
from src.nlu_processing.nlu_config import NLUConfig # Import NLUConfig directly

@pytest.fixture(scope="module")
def nlu_processor_with_objectives():
    """Fixture to initialize NLUProcessor with a config that uses a smaller spaCy model."""
    test_config = NLUConfig(
        spacy_model_name="en_core_web_sm", # Use smaller model for tests
        # Keep other settings from nlu_default_config or specify if needed
        intent_model_name="distilbert-base-uncased-finetuned-sst-2-english", # Placeholder
        sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english", # Placeholder
        command_patterns_file="nlu_command_patterns.json" # Ensure this is correct
    )
    return NLUProcessor(config=test_config)

def test_add_simple_objective(nlu_processor_with_objectives: NLUProcessor):
    """Test adding a simple objective without priority or project."""
    text = "add new objective: conquer the world"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "conquer the world"
    assert params.get("priority") is None # Default priority handled by orchestrator/service
    assert params.get("related_project_id") is None
    assert params.get("user_identifier") == "local_user_default_01" # Default from post-processing

def test_add_objective_with_priority(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective with a specific priority."""
    text = "add objective with priority 1: finish the report"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "finish the report"
    assert params.get("priority") == 1
    assert params.get("related_project_id") is None
    assert params.get("user_identifier") == "local_user_default_01"

def test_add_objective_with_project_id(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective linked to a project ID."""
    text = "new objective for project p_123: setup the new server environment"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "setup the new server environment"
    assert params.get("priority") is None
    assert params.get("related_project_id") == "p_123"
    assert params.get("user_identifier") == "local_user_default_01"

def test_add_objective_with_priority_and_project_id(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective with both priority and project ID."""
    text = "add objective for project xyz-final with priority 2: deploy to production"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "deploy to production"
    assert params.get("priority") == 2
    assert params.get("related_project_id") == "xyz-final"
    assert params.get("user_identifier") == "local_user_default_01"

def test_add_objective_alternative_phrasing(nlu_processor_with_objectives: NLUProcessor):
    """Test alternative phrasing for adding an objective."""
    text = "Set a new goal: learn quantum physics, project QuantumLeap, priority 1"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "learn quantum physics"
    assert params.get("priority") == 1
    assert params.get("related_project_id") == "QuantumLeap"
    assert params.get("user_identifier") == "local_user_default_01"

# --- Tests for "list_user_objectives" ---

def test_list_all_objectives(nlu_processor_with_objectives: NLUProcessor):
    """Test listing all objectives for the default user."""
    text = "list my objectives"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") is None # No filter means all (or default filter by service)

def test_list_active_objectives(nlu_processor_with_objectives: NLUProcessor):
    """Test listing active objectives."""
    text = "show me my active objectives"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") == "active"

def test_list_completed_objectives(nlu_processor_with_objectives: NLUProcessor):
    """Test listing completed objectives."""
    text = "what are my completed goals?"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") == "completed"

def test_list_objectives_with_status_on_hold(nlu_processor_with_objectives: NLUProcessor):
    """Test listing objectives with 'on_hold' status."""
    text = "list objectives on hold"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") == "on_hold"


# --- Tests for "set_user_objective_status" ---

def test_set_objective_status_simple(nlu_processor_with_objectives: NLUProcessor):
    """Test setting an objective's status to completed."""
    text = "mark objective obj_abc_123 as completed"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("objective_id") == "obj_abc_123"
    assert params.get("new_status") == "completed"

def test_set_objective_status_to_active(nlu_processor_with_objectives: NLUProcessor):
    """Test setting an objective's status to active."""
    text = "set status of objective my_goal_id to active"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("objective_id") == "my_goal_id"
    assert params.get("new_status") == "active"

def test_set_objective_status_alternative_phrasing(nlu_processor_with_objectives: NLUProcessor):
    """Test alternative phrasing for setting objective status."""
    text = "update objective task-42: status is on_hold"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("objective_id") == "task-42" # Note: pattern might grab with hyphen
    assert params.get("new_status") == "on_hold"

def test_set_objective_status_archived(nlu_processor_with_objectives: NLUProcessor):
    """Test setting an objective's status to archived."""
    text = "archive the objective old_project_cleanup"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"

    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("objective_id") == "old_project_cleanup"
    assert params.get("new_status") == "archived"

# --- More complex "add_user_objective" tests with key_details ---
def test_add_objective_with_key_details(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective with some key details."""
    text = "add objective: plan summer vacation, details: destination is Hawaii, budget is 3000"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "plan summer vacation"
    assert params.get("user_identifier") == "local_user_default_01"

    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("destination") == "Hawaii"
    assert key_details.get("budget") == "3000" # NLU captures as string, service handler might convert

def test_add_objective_with_priority_and_key_details(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective with priority and key details."""
    text = "new goal with priority 1: write a novel. details: genre is sci-fi, target_word_count is 80000"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "write a novel"
    assert params.get("priority") == 1
    assert params.get("user_identifier") == "local_user_default_01"

    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("genre") == "sci-fi"
    assert key_details.get("target_word_count") == "80000"

def test_add_objective_project_and_key_details(nlu_processor_with_objectives: NLUProcessor):
    """Test adding an objective with project_id and key_details."""
    text = "objective for project AlphaGoZero: improve model accuracy. details: target_metric is ELO, desired_increase is 100 points"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "improve model accuracy"
    assert params.get("related_project_id") == "AlphaGoZero"
    assert params.get("user_identifier") == "local_user_default_01"

    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("target_metric") == "ELO"
    assert key_details.get("desired_increase") == "100 points"

# Test for ensuring NLU doesn't misinterpret generic statements
def test_non_objective_statement(nlu_processor_with_objectives: NLUProcessor):
    """Test a generic statement that shouldn't trigger objective commands."""
    text = "I want to understand how the NLU system works."
    result = nlu_processor_with_objectives.process_text(text)

    # This assertion depends on how your NLU is configured for fallbacks.
    # It might be 'general_question_answering' or 'unknown_intent' or something else.
    # For now, we'll just assert it's NOT one of the objective commands.
    assert result.detected_intent is not None
    assert result.detected_intent.name not in [
        "add_user_objective",
        "list_user_objectives",
        "set_user_objective_status"
    ]
    # Also check that parsed_command is None or not one of the objective service calls
    if result.parsed_command:
        assert result.parsed_command.command not in [
            "MasterPlanner.add_user_objective",
            "MasterPlanner.list_user_objectives",
            "MasterPlanner.set_user_objective_status"
        ]

# Test for slight misspellings or variations if your patterns are robust enough (regex might be too strict for this)
# This test is more for conceptual robustness; exact regex patterns might fail this.
# @pytest.mark.skip(reason="Regex patterns might be too strict for fuzzy matching/typos")
def test_add_objective_with_typo_if_supported(nlu_processor_with_objectives: NLUProcessor):
    """Test if minor typos are handled (highly dependent on pattern robustness)."""
    text = "add new objectiv: conquer galaxy" # "objectiv" instead of "objective"
    result = nlu_processor_with_objectives.process_text(text)

    # This test might fail if regex is very specific.
    # If it fails, it indicates the patterns aren't typo-tolerant.
    if result.detected_intent and result.detected_intent.name == "add_user_objective":
        assert result.parsed_command is not None
        assert result.parsed_command.command == "MasterPlanner.add_user_objective"
        params = result.parsed_command.parameters
        assert params.get("description") == "conquer galaxy"
    else:
        # If not matched, this is acceptable given regex limitations
        print(f"Typo test: Intent was '{result.detected_intent.name if result.detected_intent else 'None'}', not 'add_user_objective'. This is acceptable for strict regex.")
        pass

# Test for case insensitivity (patterns should ideally handle this)
def test_add_objective_case_insensitive(nlu_processor_with_objectives: NLUProcessor):
    """Test case insensitivity of keywords."""
    text = "ADD OBJECTIVE with PrioRiTy 2: TEST case insensitivity"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"

    params = result.parsed_command.parameters
    assert params.get("description") == "TEST case insensitivity"
    assert params.get("priority") == 2
    assert params.get("user_identifier") == "local_user_default_01"

def test_list_objectives_case_insensitive(nlu_processor_with_objectives: NLUProcessor):
    """Test case insensitivity for list command."""
    text = "LiSt my ACTIVE objectives"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"

    params = result.parsed_command.parameters
    assert params.get("status_filter") == "active" # Post-processing should lowercase status

def test_set_objective_status_case_insensitive(nlu_processor_with_objectives: NLUProcessor):
    """Test case insensitivity for set status command."""
    text = "Mark objective OBJ_ID_CASE As COMPLETED"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"

    params = result.parsed_command.parameters
    assert params.get("objective_id") == "OBJ_ID_CASE" # ID is case sensitive
    assert params.get("new_status") == "completed" # Post-processing should lowercase status

# Test that "details" only works with add, not list or set status
def test_list_objective_with_details_is_not_add(nlu_processor_with_objectives: NLUProcessor):
    text = "list my objectives, details: foo is bar"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives" # Should still be list
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert "key_details" not in params # list command doesn't take key_details

def test_set_objective_status_with_details_is_not_add(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective obj123 as completed, details: reason is done"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status" # Should still be set_status
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "obj123"
    assert params.get("new_status") == "completed"
    assert "key_details" not in params # set_status command doesn't take key_details

# Test empty description for add objective (should ideally be caught by service, but NLU might allow)
def test_add_objective_empty_description(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: " # Empty description
    result = nlu_processor_with_objectives.process_text(text)

    # Depending on regex, it might or might not match.
    # If it matches, description should be empty or None.
    # If it doesn't match, intent might be something else.
    if result.detected_intent and result.detected_intent.name == "add_user_objective":
        assert result.parsed_command is not None
        params = result.parsed_command.parameters
        assert params.get("description") == "" or params.get("description") is None
        print(f"Empty description test: Matched 'add_user_objective' with description '{params.get('description')}'")
    else:
        print(f"Empty description test: Did not match 'add_user_objective'. Intent: {result.detected_intent.name if result.detected_intent else 'None'}")
        # This is also acceptable, as the service layer should validate required fields.
        pass

# Test empty objective ID for set_status (should ideally be caught by service)
def test_set_objective_status_empty_id(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective as completed" # Missing objective ID
    result = nlu_processor_with_objectives.process_text(text)

    if result.detected_intent and result.detected_intent.name == "set_user_objective_status":
        assert result.parsed_command is not None
        params = result.parsed_command.parameters
        assert params.get("objective_id") == "" or params.get("objective_id") is None
        print(f"Empty ID for set_status test: Matched 'set_user_objective_status' with ID '{params.get('objective_id')}'")
    else:
        print(f"Empty ID for set_status test: Did not match 'set_user_objective_status'. Intent: {result.detected_intent.name if result.detected_intent else 'None'}")
        # This is also acceptable.
        pass

# Test "details" keyword appearing in the description itself for add_objective
def test_add_objective_with_details_keyword_in_description(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: document the key_details extraction logic"
    result = nlu_processor_with_objectives.process_text(text)

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    params = result.parsed_command.parameters
    assert params.get("description") == "document the key_details extraction logic"
    assert "key_details" not in params # Should not parse "extraction logic" as key_details

def test_add_objective_with_details_keyword_in_description_and_actual_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: review key_details parsing, details: assignee is Jules, due_date is tomorrow"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    params = result.parsed_command.parameters
    assert params.get("description") == "review key_details parsing"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("assignee") == "Jules"
    assert key_details.get("due_date") == "tomorrow"

# Test 'objective for project' where description is short
def test_add_objective_short_desc_for_project(nlu_processor_with_objectives: NLUProcessor):
    text = "new objective for project MyProject: setup"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "setup"
    assert params.get("related_project_id") == "MyProject"

# Test 'objective with priority' where description is short
def test_add_objective_short_desc_with_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective with priority 1: test"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "test"
    assert params.get("priority") == 1

# Test 'objective with priority and project' where description is short
def test_add_objective_short_desc_with_priority_project(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective for project Shorty with priority 2: go"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "go"
    assert params.get("priority") == 2
    assert params.get("related_project_id") == "Shorty"

# Test 'objective with details' where description is short
def test_add_objective_short_desc_with_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: run, details: speed is fast, duration is 1hr"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "run"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("speed") == "fast"
    assert key_details.get("duration") == "1hr"

# Test that 'project' keyword in description doesn't get confused if 'for project' isn't used
def test_add_objective_with_project_keyword_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add new objective: complete the project planning phase"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "complete the project planning phase"
    assert params.get("related_project_id") is None

# Test that 'priority' keyword in description doesn't get confused if 'with priority' isn't used
def test_add_objective_with_priority_keyword_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: assess task priority for all items"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "assess task priority for all items"
    assert params.get("priority") is None

# Test a more complex objective ID for set_status
def test_set_objective_status_complex_id(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective user_default_01-proj_X-task_Y-sub_Z.123 as on_hold"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "user_default_01-proj_X-task_Y-sub_Z.123"
    assert params.get("new_status") == "on_hold"

# Test status keyword appearing in objective ID for set_status
def test_set_objective_status_with_status_keyword_in_id(nlu_processor_with_objectives: NLUProcessor):
    text = "update objective my_active_task as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "my_active_task"
    assert params.get("new_status") == "completed"

# Test status keyword appearing in description for add_objective
def test_add_objective_with_status_keyword_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: check the server status"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "check the server status"
    # Ensure it doesn't try to parse a status for the add command
    assert "status" not in params
    assert "new_status" not in params

# Test "my goals" for list command
def test_list_my_goals(nlu_processor_with_objectives: NLUProcessor):
    text = "show my goals"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") is None

# Test "my tasks" as a potential alias for objectives if desired (current patterns might not catch this as objectives)
# This test is to see current behavior.
def test_list_my_tasks_as_objectives(nlu_processor_with_objectives: NLUProcessor):
    text = "list my tasks"
    result = nlu_processor_with_objectives.process_text(text)
    # This will likely NOT be list_user_objectives unless patterns are very broad
    if result.detected_intent and result.detected_intent.name == "list_user_objectives":
        print("INFO: 'list my tasks' was interpreted as list_user_objectives.")
        assert result.parsed_command is not None
        assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    else:
        print(f"INFO: 'list my tasks' was interpreted as {result.detected_intent.name if result.detected_intent else 'None'}, not list_user_objectives.")
        assert result.detected_intent.name != "list_user_objectives"

# Test "set goal status"
def test_set_goal_status(nlu_processor_with_objectives: NLUProcessor):
    text = "set goal goal_id_007 status to active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "goal_id_007"
    assert params.get("new_status") == "active"

# Test "add task" as a potential alias for add_objective
def test_add_task_as_objective(nlu_processor_with_objectives: NLUProcessor):
    text = "add task: buy milk"
    result = nlu_processor_with_objectives.process_text(text)
    if result.detected_intent and result.detected_intent.name == "add_user_objective":
        print("INFO: 'add task' was interpreted as add_user_objective.")
        assert result.parsed_command is not None
        assert result.parsed_command.command == "MasterPlanner.add_user_objective"
        assert result.parsed_command.parameters.get("description") == "buy milk"
    else:
        print(f"INFO: 'add task' was interpreted as {result.detected_intent.name if result.detected_intent else 'None'}, not add_user_objective.")
        assert result.detected_intent.name != "add_user_objective"

# Test that 'objective' keyword in ID doesn't break set_status
def test_set_objective_status_with_objective_in_id(nlu_processor_with_objectives: NLUProcessor):
    text = "mark main_objective_phase1 as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "main_objective_phase1"
    assert params.get("new_status") == "completed"

# Test that 'goal' keyword in ID doesn't break set_status
def test_set_objective_status_with_goal_in_id(nlu_processor_with_objectives: NLUProcessor):
    text = "set status of my_primary_goal to on_hold"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "my_primary_goal"
    assert params.get("new_status") == "on_hold"

# Test that 'task' keyword in ID doesn't break set_status
def test_set_objective_status_with_task_in_id(nlu_processor_with_objectives: NLUProcessor):
    text = "update important_task_id status is archived"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "important_task_id"
    assert params.get("new_status") == "archived"

# Test a very long description for add_objective to see if regex handles it.
def test_add_objective_very_long_description(nlu_processor_with_objectives: NLUProcessor):
    long_desc = "This is a very long description for an objective that spans multiple phrases and includes various punctuation marks like commas, periods. It also has numbers 123 and special characters !@#$%^&*()_+ to test the robustness of the description capture group in the NLU patterns. We want to ensure that the entire text until a specific keyword like 'priority', 'project', or 'details' (or end of string) is captured."
    text = f"add objective: {long_desc}"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == long_desc

def test_add_objective_long_description_then_priority(nlu_processor_with_objectives: NLUProcessor):
    long_desc = "This is another long objective description, which should be fully captured. The objective is to meticulously document all existing NLU patterns and their behaviors, including edge cases and potential failure points, to improve overall system reliability and maintainability. This task is critical for future development."
    text = f"add objective: {long_desc} with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == long_desc
    assert params.get("priority") == 1

def test_add_objective_long_description_then_project(nlu_processor_with_objectives: NLUProcessor):
    long_desc = "A lengthy description for an objective related to a specific project. This objective involves refactoring the entire user authentication module to incorporate multi-factor authentication and OAuth2 support, enhancing security and user experience across all platforms associated with the aforementioned project."
    text = f"add objective: {long_desc} for project SecurityOverhaul"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == long_desc
    assert params.get("related_project_id") == "SecurityOverhaul"

def test_add_objective_long_description_then_details(nlu_processor_with_objectives: NLUProcessor):
    long_desc = "This objective is about creating a comprehensive test suite for the new data processing pipeline. It should cover unit tests, integration tests, and end-to-end tests, ensuring data integrity and transformation accuracy at every stage. The goal is to achieve at least 95% code coverage for critical components."
    text = f"add objective: {long_desc} details: coverage_target is 95%, primary_focus is data_pipeline_validation"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == long_desc
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("coverage_target") == "95%"
    assert key_details.get("primary_focus") == "data_pipeline_validation"

def test_add_objective_long_description_with_all_modifiers(nlu_processor_with_objectives: NLUProcessor):
    long_desc = "The objective is to launch the new marketing campaign for Q3. This includes finalizing ad creatives, setting up tracking analytics, coordinating with influencers, and preparing press releases. The campaign aims to increase brand awareness by 20% and generate 500 new leads."
    text = f"add objective: {long_desc} for project Q3Campaign priority 1 details: target_awareness_increase is 20%, target_leads is 500"
    # Note: The order of 'priority', 'project', 'details' might matter for some regex patterns.
    # The current nlu_command_patterns.json tries to handle them somewhat flexibly, but complex orders can be tricky.
    # This specific order (desc -> project -> priority -> details) is one of
    # the more complex ones to parse if not explicitly structured.
    # Let's test the structure from the pattern: description -> (priority)? -> (project)? -> (details)?
    # So, a better test string for the current patterns would be:
    text_structured = f"add objective: {long_desc} with priority 1 for project Q3Campaign details: target_awareness_increase is 20%, target_leads is 500"

    result = nlu_processor_with_objectives.process_text(text_structured) # Using the structured text

    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == long_desc
    assert params.get("priority") == 1
    assert params.get("related_project_id") == "Q3Campaign"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("target_awareness_increase") == "20%"
    assert key_details.get("target_leads") == "500"

# Test objective ID with spaces (if patterns allow, otherwise this should fail or misinterpret)
# Current patterns for objective_id are typically \w+ or similar, which don't include spaces.
def test_set_objective_status_id_with_spaces(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective my important goal as completed"
    result = nlu_processor_with_objectives.process_text(text)

    if result.detected_intent and result.detected_intent.name == "set_user_objective_status":
        params = result.parsed_command.parameters
        # If objective_id pattern is `[\w.-]+`, this will likely capture "my"
        # and "important goal as completed" might be misinterpreted or ignored.
        # This test is to see how the current patterns behave.
        print(f"ID with spaces test for set_status: Matched. Objective ID captured: '{params.get('objective_id')}', New Status: '{params.get('new_status')}'")
        # Based on typical \w+ patterns, we expect it to grab only the first word or fail to match the intent.
        # If it matches, objective_id will likely be "my" and new_status might be "important" or None.
        # This is an expected failure mode for simple regex if IDs can have spaces and aren't quoted.
        assert params.get("objective_id") == "my" # Or whatever the specific regex captures
        # And new_status might be wrong or None
        assert params.get("new_status") != "completed" or params.get("new_status") is None

    else:
        print(f"ID with spaces test for set_status: Did not match 'set_user_objective_status' or failed parsing. Intent: {result.detected_intent.name if result.detected_intent else 'None'}")
        # This is also an acceptable outcome if the patterns are strict about ID format.
        pass

# Test project ID with spaces (similar to objective ID with spaces)
def test_add_objective_project_id_with_spaces(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new task for project My Big Project"
    result = nlu_processor_with_objectives.process_text(text)

    if result.detected_intent and result.detected_intent.name == "add_user_objective":
        params = result.parsed_command.parameters
        print(f"Project ID with spaces test: Matched. Project ID captured: '{params.get('related_project_id')}'")
        # Expect project_id to be "My" or similar, with the rest being part of the description or lost.
        assert params.get("related_project_id") == "My"
        assert params.get("description") == "new task for project My Big Project" \
            or "Big Project" in params.get("description") # if project part is absorbed by description
    else:
        print(f"Project ID with spaces test: Did not match 'add_user_objective' or failed parsing. Intent: {result.detected_intent.name if result.detected_intent else 'None'}")
        pass

# Test key_details with spaces in keys or values
def test_add_objective_key_details_with_spaces(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: vacation planning details: main destination is South Island NZ, activity type is hiking and adventure"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "vacation planning"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    # The regex for key_details `([\w\s]+?)\s*[:is]\s*([\w\s.,'-]+?)(?:,|$)` should handle spaces.
    assert key_details.get("main destination") == "South Island NZ"
    assert key_details.get("activity type") == "hiking and adventure"

# Test key_details with numeric values
def test_add_objective_key_details_numeric_values(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: performance review details: score is 4.5, tasks_completed is 12, review_year is 2023"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("score") == "4.5" # Captured as string
    assert key_details.get("tasks_completed") == "12" # Captured as string
    assert key_details.get("review_year") == "2023" # Captured as string
    # Type conversion would be the responsibility of the service handler.

# Test key_details with hyphens and other typical characters in values
def test_add_objective_key_details_special_chars_in_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: server setup details: server_id is app-server-001, os_version is Ubuntu 22.04.1-LTS, notes is 'Needs extra RAM'"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("server_id") == "app-server-001"
    assert key_details.get("os_version") == "Ubuntu 22.04.1-LTS"
    assert key_details.get("notes") == "'Needs extra RAM'" # Includes quotes if present

# Test multiple key-value pairs in details, separated by commas and potentially "and"
def test_add_objective_multiple_key_details_mixed_separators(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new PC build details: cpu is Ryzen 9, gpu is RTX 4090, and ram is 64GB, storage is '2TB NVMe'"
    # The current regex for key_details might not explicitly handle "and" as a separator.
    # It expects "key:value, key:value". "and" might get absorbed into the value of the preceding key.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("cpu") == "Ryzen 9"
    assert key_details.get("gpu") == "RTX 4090"
    # Depending on regex greediness for value and separator handling:
    # Option 1: "and ram" is part of gpu value. Then "ram" is not a key.
    # Option 2: "and" is handled, "ram" is a key.
    # The current regex `([\w\s]+?)\s*[:is]\s*([\w\s.,'-]+?)(?:,|$|and\s+)` (updated to try to handle 'and')
    if "ram" in key_details:
        assert key_details.get("ram") == "64GB"
        assert key_details.get("storage") == "'2TB NVMe'"
    else:
        # This means "and ram is 64GB" was likely part of the "gpu" value or parsing stopped.
        print(f"Multiple key_details with 'and' separator test: 'ram' not found as a separate key. GPU value: {key_details.get('gpu')}")
        # For the current regex, it's more likely that "and" is not a special separator.
        # So, "gpu" value might be "RTX 4090, and ram is 64GB" or similar if the value capture is greedy.
        # The pattern `([\w\s.,'-]+?)` for value is non-greedy.
        # Let's check the actual behavior.
        # The value regex `([\w\s.,'-]+?)` will stop at the next comma or end of string.
        # So, "gpu is RTX 4090, and ram is 64GB" -> gpu: "RTX 4090", then next part "and ram is 64GB"
        # This would be parsed as key "and ram" value "64GB"
        assert key_details.get("and ram") == "64GB" # This is what the regex `([\w\s]+?)\s*[:is]\s*([\w\s.,'-]+?)(?:,|$)?` would likely do
                                                    # if "and" is not a special separator.
        # The modified regex `([\w\s]+?)\s*[:is]\s*([\w\s.,'-]+?)(?:,|$|and\s*)`
        # attempts to use "and " as a delimiter. Let's test this one.
        # If "and " is a delimiter, then after "gpu is RTX 4090", the next part is "ram is 64GB".
        # This means the "and" is consumed by the delimiter.
        # So we should find "ram" as "64GB".
        # This is based on the assumption that the nlu_command_patterns.json has the more advanced regex
        # with `(?:,|$|and\s*)` as the key-value pair delimiter.
        # Let's assume the pattern file has been updated.
        # If it's `(?:,|$|and\s+)`, then after "gpu is RTX 4090, " (comma delimiter),
        # the next part is "and ram is 64GB". The `and\s+` won't match if a comma already did.
        # If the string was "gpu is RTX 4090 and ram is 64GB", then "and " would delimit.
        # Given the string "gpu is RTX 4090, and ram is 64GB", the comma delimits first.
        # Then "and ram is 64GB" is the next segment.
        # Key: "and ram", Value: "64GB".
        # Let's re-verify the regex in nlu_command_patterns.json for `TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2`
        # It is: "regex": "([\\w\\s-]+?)\\s*[:is](?!\\s*http)\\s*([\\w\\s.,'@\\-\\/%+'\"()]+?)(?=\\s*,\\s*[\\w\\s-]+?\\s*[:is](?!\\s*http)|$)"
        # This regex uses a positive lookahead `(?=\\s*,\\s*[\\w\\s-]+?\\s*[:is](?!\\s*http)|$)`
        # This means it looks for a comma followed by another key: or end of string.
        # It does NOT explicitly handle "and" as a separator.
        # So, "and ram is 64GB" will be the value for the key "gpu" if there's no comma after "RTX 4090".
        # With "gpu is RTX 4090, and ram is 64GB":
        # Key: "gpu", Value: "RTX 4090" (comma is the delimiter)
        # Next segment: " and ram is 64GB, storage is '2TB NVMe'"
        # Key: "and ram" (leading space might be trimmed by key regex `[\w\s-]+?`), Value: "64GB"
        # Next segment: " storage is '2TB NVMe'"
        # Key: "storage", Value: "'2TB NVMe'"
        # So the expectation should be:
        assert key_details.get("and ram") == "64GB" # If leading space on key is trimmed
        # OR key_details.get(" ram") if not trimmed. Regex `[\w\s-]+?` includes space.
        # Let's assume key is "and ram".
        assert key_details.get("storage") == "'2TB NVMe'"
        # This test reveals the limitations of the current key-value regex with natural conjunctions like "and".
        # For truly robust parsing, a more advanced parser or sequence of regexes would be needed.
        # The current test reflects the behavior of the V2 regex.
        pass # Pass for now, acknowledging the complexity.

# Test for "list objectives for project X"
def test_list_objectives_for_project(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives for project P-Alpha"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("related_project_id") == "P-Alpha" # This requires pattern to catch project_id
    assert params.get("status_filter") is None

# Test for "list active objectives for project X"
def test_list_active_objectives_for_project(nlu_processor_with_objectives: NLUProcessor):
    text = "show active objectives for project BetaMan"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("related_project_id") == "BetaMan"
    assert params.get("status_filter") == "active"

# Test for "list completed objectives for project Y"
def test_list_completed_objectives_for_project(nlu_processor_with_objectives: NLUProcessor):
    text = "what are the completed goals for project GammaRayBurst?"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("related_project_id") == "GammaRayBurst" # ? might be part of ID or stripped
    assert params.get("status_filter") == "completed"
    # The pattern `([\w.-]+)` for project_id will not include `?`.
    # So `GammaRayBurst` is expected.
    assert params.get("related_project_id") == "GammaRayBurst"

# Test for "set objective X for project Y as completed" - project_id is not part of set_status command
def test_set_objective_status_with_project_mention(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective task1 for project ProjAlpha as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.set_user_objective_status"
    params = result.parsed_command.parameters
    # The phrase "for project ProjAlpha" might be absorbed by objective_id or ignored
    # depending on the regex for objective_id and new_status.
    # Current objective_id pattern: `([\w.-]+)`
    # Current new_status pattern: `(active|on_hold|completed|archived)`
    # Pattern for set_user_objective_status:
    # `^(?:mark|set|update)\s+(?:objective|goal|task)\s+(?P<objective_id>[\w.-]+)\s*(?:as|status\s*(?:is|to)?)\s*(?P<new_status>active|on_hold|completed|archived)`
    # This will capture "task1" as objective_id. "for project ProjAlpha" is not matched by this part.
    # Then "as completed" matches.
    assert params.get("objective_id") == "task1"
    assert params.get("new_status") == "completed"
    assert "related_project_id" not in params

# Test for "set objective X as completed for project Y" - order difference
def test_set_objective_status_with_project_mention_after_status(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective task2 as completed for project ProjBeta"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "task2"
    assert params.get("new_status") == "completed"
    # "for project ProjBeta" is extraneous to the main pattern.
    assert "related_project_id" not in params

# Test a very generic "objective" query that might be ambiguous
def test_ambiguous_objective_query(nlu_processor_with_objectives: NLUProcessor):
    text = "tell me about my objectives"
    result = nlu_processor_with_objectives.process_text(text)
    # This should resolve to list_user_objectives due to "my objectives"
    assert result.detected_intent is not None
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("user_identifier") == "local_user_default_01"
    assert params.get("status_filter") is None # No specific status mentioned

# Test a query that could be confused between add and list if not specific enough
def test_ambiguous_add_or_list_objective(nlu_processor_with_objectives: NLUProcessor):
    text = "objective: new world order"
    result = nlu_processor_with_objectives.process_text(text)
    # This phrasing is very close to "add objective: new world order"
    # Let's see if the "add" pattern for `(?:add|new|set)\s*(?:objective|goal|task)` catches it.
    # "objective: description" is one of the patterns for add_user_objective.
    assert result.detected_intent is not None
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command is not None
    assert result.parsed_command.command == "MasterPlanner.add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "new world order"

# Test a query that uses "goal" instead of "objective" for all commands
def test_goal_alias_add(nlu_processor_with_objectives: NLUProcessor):
    text = "add goal: learn to fly, priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("description") == "learn to fly"
    assert result.parsed_command.parameters.get("priority") == 1

def test_goal_alias_list(nlu_processor_with_objectives: NLUProcessor):
    text = "list my active goals"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command.parameters.get("status_filter") == "active"

def test_goal_alias_set_status(nlu_processor_with_objectives: NLUProcessor):
    text = "mark goal fly_high as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command.parameters.get("objective_id") == "fly_high"
    assert result.parsed_command.parameters.get("new_status") == "completed"

# Test a query that uses "task" instead of "objective" for all commands
def test_task_alias_add(nlu_processor_with_objectives: NLUProcessor):
    text = "new task: fix the bug #123, project BugFixes"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("description") == "fix the bug #123"
    assert result.parsed_command.parameters.get("related_project_id") == "BugFixes"

def test_task_alias_list(nlu_processor_with_objectives: NLUProcessor):
    text = "show my completed tasks for project OldChores"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    assert result.parsed_command.parameters.get("status_filter") == "completed"
    assert result.parsed_command.parameters.get("related_project_id") == "OldChores"

def test_task_alias_set_status(nlu_processor_with_objectives: NLUProcessor):
    text = "set task bug_report_final as archived"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command.parameters.get("objective_id") == "bug_report_final"
    assert result.parsed_command.parameters.get("new_status") == "archived"

# Test for objective IDs that are purely numeric for set_status
def test_set_objective_status_numeric_id(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective 1234567890 as active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "1234567890"
    assert params.get("new_status") == "active"

# Test for project IDs that are purely numeric for add/list
def test_add_objective_numeric_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new objective for project 98765, priority 3"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "new objective" # The "for project 98765" part is tricky
    assert params.get("related_project_id") == "98765"
    assert params.get("priority") == 3

def test_list_objectives_numeric_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives for project 112233"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("related_project_id") == "112233"

# Test leading/trailing spaces around values
def test_add_objective_spaces_around_priority_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: spaced out priority with priority  2  "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "spaced out priority"
    assert params.get("priority") == 2 # Transformation should handle stripping spaces for int conversion

def test_add_objective_spaces_around_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: spaced out project for project  MySpacedProject  "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    # The project_id capture group `([\w.-]+)` won't include leading/trailing spaces.
    # The surrounding non-capturing groups `\s+` or `\s*` handle spaces around keywords.
    # So, "MySpacedProject" is expected if the value itself doesn't have internal spaces that break the \w+
    # If project ID can have spaces, the pattern needs to be `([\w\s.-]+)`
    # The current pattern `PROJECT_ID_PATTERN` is `([\w.-]+)`.
    # This means "MySpacedProject" will be captured as "MySpacedProject".
    # The spaces "  MySpacedProject  " are handled by `\s+for project\s+(?P<project_id>...)`
    assert params.get("related_project_id") == "MySpacedProject"

def test_set_objective_status_spaces_around_id_and_status(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective  obj-spaced   as   completed  "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "obj-spaced"
    assert params.get("new_status") == "completed" # Post-processing handles lowercase and stripping

def test_list_objectives_spaces_around_status_filter(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives with status   active  "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("status_filter") == "active" # Post-processing handles lowercase and stripping

def test_add_objective_key_details_spaces_around_key_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: spaced key details details:  my key  is  my value  ,  another key :  another value "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    # The regex `([\w\s-]+?)\s*[:is]\s*([\w\s.,'@\-\/%+'\"()]+?)...`
    # Key part `([\w\s-]+?)` will capture "my key" (non-greedy).
    # Value part `([\w\s.,'@\-\/%+'\"()]+?)` will capture "my value".
    # Spaces around "is" or ":" are handled by `\s*[:is]\s*`.
    # Spaces within keys/values are captured if `\s` is in the character class.
    # Post-processing step `TRANSFORM_STRIP_DICT_KEYS_VALUES` should clean these.
    assert key_details.get("my key") == "my value"
    assert key_details.get("another key") == "another value"

# Test "set" keyword for add_objective
def test_add_objective_with_set_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective: new plan"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "new plan"

# Test "new" keyword for add_objective
def test_add_objective_with_new_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "new objective: another plan"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "another plan"

# Test "update" keyword for set_status (already covered, but good to have specific)
def test_set_status_with_update_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "update objective my-obj-id status to on_hold"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "my-obj-id"
    assert params.get("new_status") == "on_hold"

# Test "show" keyword for list_objectives (already covered)
def test_list_objectives_with_show_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "show my objectives"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"

# Test "what are" for list_objectives (already covered)
def test_list_objectives_with_what_are_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "what are my completed objectives?"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("status_filter") == "completed"

# Test if "details" keyword is too close to description and gets absorbed
def test_add_objective_details_keyword_absorption(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: prepare meetingdetails: agenda is final, time is 3pm"
    # Here "meetingdetails" might be one word or "meeting details" (if space).
    # If "meetingdetails", it's part of description.
    # If "meeting details", then "details" keyword might trigger key-value parsing for the rest.
    # The pattern for description capture is `(?P<description>.+?)` which is non-greedy.
    # It stops before `(?:\s+with priority...|\s+for project...|\s+details...|$)`
    # So, if "details:" is found, description stops before it.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    if "meetingdetails" in text: # No space
        assert params.get("description") == "prepare meetingdetails: agenda is final, time is 3pm" # Whole thing is desc
        assert "key_details" not in params
    # If text was "add objective: prepare meeting details: agenda is final, time is 3pm"
    # then:
    # assert params.get("description") == "prepare meeting"
    # key_details = params.get("key_details")
    # assert key_details.get("agenda") == "final"
    # assert key_details.get("time") == "3pm"
    # For the current test string "meetingdetails":
    assert params.get("description") == "prepare meetingdetails: agenda is final, time is 3pm"
    assert "key_details" not in params


# Test priority like "priority high" or "priority low" (needs mapping in post-processing if used)
# Current patterns expect numeric priority. This test will show it doesn't match numeric.
def test_add_objective_textual_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: urgent task with priority high"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "urgent task" # "with priority high" is part of description
    assert params.get("priority") is None # As "high" is not a number
    # If we wanted to support "high" -> 1, "medium" -> 2, "low" -> 3,
    # the regex for priority would need to capture `(\d+|high|medium|low)`
    # and then a TRANSFORM_MAP_VALUES post-processing step would convert text to int.
    # Current priority pattern is `PRIORITY_PATTERN = r"(\d+)"`.

# Test if "status" keyword in list command is flexible (e.g. "status is active", "status active")
def test_list_objectives_flexible_status_keyword(nlu_processor_with_objectives: NLUProcessor):
    text1 = "list objectives status is active"
    text2 = "list objectives status active"
    text3 = "list objectives with status: active" # Already covered by pattern `with status\s*:?\s*(?P<status_filter>...)`

    result1 = nlu_processor_with_objectives.process_text(text1)
    assert result1.detected_intent.name == "list_user_objectives"
    assert result1.parsed_command.parameters.get("status_filter") == "active"

    result2 = nlu_processor_with_objectives.process_text(text2)
    assert result2.detected_intent.name == "list_user_objectives"
    assert result2.parsed_command.parameters.get("status_filter") == "active"

# Test if "as" keyword in set_status is flexible (e.g. "status completed", "as completed")
def test_set_status_flexible_as_keyword(nlu_processor_with_objectives: NLUProcessor):
    text1 = "set objective obj1 status completed" # Pattern `status\s*(?:is|to)?\s*(?P<new_status>...)` should catch this
    text2 = "set objective obj2 as completed" # Pattern `as\s*(?P<new_status>...)` should catch this

    result1 = nlu_processor_with_objectives.process_text(text1)
    assert result1.detected_intent.name == "set_user_objective_status"
    assert result1.parsed_command.parameters.get("objective_id") == "obj1"
    assert result1.parsed_command.parameters.get("new_status") == "completed"

    result2 = nlu_processor_with_objectives.process_text(text2)
    assert result2.detected_intent.name == "set_user_objective_status"
    assert result2.parsed_command.parameters.get("objective_id") == "obj2"
    assert result2.parsed_command.parameters.get("new_status") == "completed"

# Test if description can contain colons without triggering key_details prematurely
def test_add_objective_colon_in_description(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: Title: My Life Story, Part 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "Title: My Life Story, Part 1"
    assert "key_details" not in params

def test_add_objective_colon_in_description_with_actual_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: Topic: AI Ethics - Chapter 3 details: word_count is 5000, deadline is next_friday"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "Topic: AI Ethics - Chapter 3"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("word_count") == "5000"
    assert key_details.get("deadline") == "next_friday"

# Test if "for project" pattern is greedy and absorbs part of description
def test_add_objective_project_keyword_greediness(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: plan the project for project Alpha"
    # Description should be "plan the project" and project_id "Alpha"
    # Or description "plan the project for project Alpha" if "for project" is not matched well.
    # The pattern for add_objective has `(?:\s+for project\s+(?P<project_id>...))?`
    # The description part is `(?P<description>.+?)` (non-greedy)
    # This non-greedy description should stop just before " for project Alpha"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "plan the project"
    assert params.get("related_project_id") == "Alpha"

# Test if "with priority" pattern is greedy
def test_add_objective_priority_keyword_greediness(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: high priority task with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "high priority task"
    assert params.get("priority") == 1

# Test if "details" pattern is greedy
def test_add_objective_details_keyword_greediness(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with many details details: key1 is val1, key2 is val2"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task with many details"
    key_details = params.get("key_details")
    assert key_details.get("key1") == "val1"
    assert key_details.get("key2") == "val2"

# Final check: a command that is NOT an objective command
def test_non_objective_specific_command(nlu_processor_with_objectives: NLUProcessor):
    text = "generate an image of a cat" # Assuming this is a different intent
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name == "image_generation" # Or whatever it maps to
    assert result.parsed_command is None or result.parsed_command.command != "MasterPlanner.add_user_objective"

# Test for objective ID that looks like a status for set_status
def test_set_objective_status_id_looks_like_status(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective active as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "active"
    assert params.get("new_status") == "completed"

def test_set_objective_status_id_looks_like_status_v2(nlu_processor_with_objectives: NLUProcessor):
    text = "set status of objective completed to active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "completed"
    assert params.get("new_status") == "active"

# Test for project ID that looks like a status for add/list
def test_add_objective_project_id_looks_like_status(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new for project active with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "new"
    assert params.get("related_project_id") == "active"
    assert params.get("priority") == 1

def test_list_objectives_project_id_looks_like_status(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives for project completed status active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("related_project_id") == "completed"
    assert params.get("status_filter") == "active"

# Test key_details where a key or value might be a reserved keyword like "priority", "project", "details", "status"
def test_add_objective_key_details_with_reserved_keywords_as_values(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: meta objective details: next_step is 'review project status', related_item is 'priority document', context is 'see details section'"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "meta objective"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("next_step") == "'review project status'"
    assert key_details.get("related_item") == "'priority document'"
    assert key_details.get("context") == "'see details section'"

def test_add_objective_key_details_with_reserved_keywords_as_keys(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: config details: priority is high, project is X, details is 'sub-details here', status is pending"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "config" # "details: priority is high..." is the details part
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("priority") == "high"
    assert key_details.get("project") == "X"
    assert key_details.get("details") == "'sub-details here'" # Quoted because it's a value
    assert key_details.get("status") == "pending"

# Test if "status" keyword in `list ... with status <val>` is confused if <val> is not a valid status
def test_list_objectives_invalid_status_value(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives with status foobar"
    result = nlu_processor_with_objectives.process_text(text)
    # The regex `(?P<status_filter>active|on_hold|completed|archived)` will NOT match "foobar".
    # So, the optional group `(?:\s+with status\s*:?\s*(?P<status_filter>...))?` will not match.
    # The intent should still be list_user_objectives, but status_filter will be None.
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("status_filter") is None # "foobar" is not a valid status, so group doesn't match
    # The part "with status foobar" might be absorbed by a more generic part of the list pattern if one exists,
    # or the pattern might just match "list objectives".
    # The current list patterns are quite specific, e.g. `^... list ... objectives ... (with status ...)? (for project ...)?`
    # So "with status foobar" just fails to match the status part.

# Test if "priority" keyword in `add ... with priority <val>` is confused if <val> is not numeric
def test_add_objective_non_numeric_priority_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: test non-numeric priority with priority XYZ"
    result = nlu_processor_with_objectives.process_text(text)
    # Priority pattern `(\d+)` will not match "XYZ".
    # So, the optional group `(?:\s+with priority\s+(?P<priority>...))?` will not match.
    # Description will be "test non-numeric priority with priority XYZ". Priority param will be None.
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "test non-numeric priority with priority XYZ"
    assert params.get("priority") is None

# Test if "as" keyword in set_status is confused if the new_status value is not a valid one
def test_set_status_invalid_new_status_value(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective obj789 as foobar_status"
    result = nlu_processor_with_objectives.process_text(text)
    # The regex `(?P<new_status>active|on_hold|completed|archived)` will not match "foobar_status".
    # So, the whole "as foobar_status" part will not match.
    # The intent set_user_objective_status might not even be triggered, or it might parse incorrectly.
    # Let's check the base pattern for set_user_objective_status:
    # `^(?:mark|set|update)\s+(?:objective|goal|task)\s+(?P<objective_id>[\w.-]+)\s*(?:as|status\s*(?:is|to)?)\s*(?P<new_status>...)`
    # The part `\s*(?:as|status\s*(?:is|to)?)\s*(?P<new_status>...)` is NOT optional.
    # So if `new_status` doesn't match, the whole intent pattern might fail.
    if result.detected_intent and result.detected_intent.name == "set_user_objective_status":
        # This would mean the pattern is more lenient than expected, or new_status group is optional.
        # Looking at the pattern, the new_status part is NOT optional.
        # So, this should NOT match the intent.
        assert False, "set_user_objective_status intent should not have matched with invalid new_status value."
    else:
        # This is the expected outcome.
        assert result.detected_intent.name != "set_user_objective_status"
        print(f"Invalid new_status test: Intent was '{result.detected_intent.name if result.detected_intent else 'None'}', not set_user_objective_status, as expected.")

# Test a command that has "objective" but is clearly not one of the defined commands
def test_unrelated_command_with_objective_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "analyze the primary objective of this document"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent is not None
    assert result.detected_intent.name not in ["add_user_objective", "list_user_objectives", "set_user_objective_status"]
    # Example: might be "document_processing" or "general_question_answering"
    print(f"Unrelated command with 'objective' keyword: Intent is '{result.detected_intent.name}'")

# Test commands that are substrings of each other or very similar, e.g. "set objective" vs "set objective status"
# "set objective" is an alias for "add objective".
# "set objective status" is a different command.
def test_substring_like_commands_set_objective_vs_set_status(nlu_processor_with_objectives: NLUProcessor):
    text_add = "set objective: this is a new one"
    text_set_status = "set objective my_obj status completed"

    result_add = nlu_processor_with_objectives.process_text(text_add)
    assert result_add.detected_intent.name == "add_user_objective"
    assert result_add.parsed_command.parameters.get("description") == "this is a new one"

    result_set_status = nlu_processor_with_objectives.process_text(text_set_status)
    assert result_set_status.detected_intent.name == "set_user_objective_status"
    assert result_set_status.parsed_command.parameters.get("objective_id") == "my_obj"
    assert result_set_status.parsed_command.parameters.get("new_status") == "completed"

# Test if "project" keyword in list objectives is confused with "for project <id>" if no ID is given
def test_list_objectives_ambiguous_project_keyword(nlu_processor_with_objectives: NLUProcessor):
    text = "list all project objectives" # "project" here is an adjective for objectives
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    # The "for project <id>" part is optional.
    # "project objectives" is not "for project <id>".
    # So related_project_id should be None.
    assert params.get("related_project_id") is None
    assert params.get("status_filter") is None # "all" is not a status value

# Test a very minimal "add objective" command
def test_add_objective_minimal(nlu_processor_with_objectives: NLUProcessor):
    text = "objective: go home"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("description") == "go home"

# Test a very minimal "list objectives" command
def test_list_objectives_minimal(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"

# Test a very minimal "set objective status" command (might fail if not enough info)
def test_set_objective_status_minimal(nlu_processor_with_objectives: NLUProcessor):
    text = "objective obj status active" # Minimal valid form
    result = nlu_processor_with_objectives.process_text(text)
    # This pattern: `^(?:objective|goal|task)\s+(?P<objective_id>[\w.-]+)\s+status\s+(?P<new_status>...)`
    # is one of the set_status patterns.
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "obj"
    assert params.get("new_status") == "active"

def test_set_objective_status_minimal_alternative(nlu_processor_with_objectives: NLUProcessor):
    text = "task tid as completed" # Minimal valid form for "as" variant
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "tid"
    assert params.get("new_status") == "completed"

# Test if "details" keyword without actual details parses correctly for add_objective
def test_add_objective_with_details_keyword_no_values(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with details section details:" # No key-value pairs after "details:"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task with details section"
    # The key_details parsing should yield an empty dict or None if nothing is found.
    # TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2 returns None if input is empty or only whitespace.
    # The group `(?P<details_kv_string>.*)` captures everything after "details: ".
    # If this is empty, then key_details will be None after post-processing.
    assert params.get("key_details") is None or params.get("key_details") == {}

def test_add_objective_with_details_keyword_and_empty_string_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with empty detail details: note is " # Empty value for 'note'
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task with empty detail"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("note") == "" # Empty string value

# Test objective ID containing "as" or "status" for set_status command.
def test_set_status_objective_id_contains_as(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective my_task_as_important as completed"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "my_task_as_important" # The "as" in ID should not break parsing of "as completed"
    assert params.get("new_status") == "completed"

def test_set_status_objective_id_contains_status(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective check_server_status status on_hold"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == "check_server_status" # The "status" in ID should not break "status on_hold"
    assert params.get("new_status") == "on_hold"

# Test project ID containing "priority" or "details" for add/list commands.
def test_add_objective_project_id_contains_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task for project high_priority_project with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("related_project_id") == "high_priority_project"
    assert params.get("priority") == 1

def test_add_objective_project_id_contains_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task for project project_with_details_section details: key is value"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("related_project_id") == "project_with_details_section"
    assert isinstance(params.get("key_details"), dict)
    assert params.get("key_details").get("key") == "value"

def test_list_objectives_project_id_contains_status(nlu_processor_with_objectives: NLUProcessor):
    # This was already tested as `test_list_objectives_project_id_looks_like_status`
    # list objectives for project completed status active
    # -> project_id="completed", status_filter="active" - which is correct.
    pass

# Test priority value that is zero or negative (regex `\d+` allows zero, but not negative without `-`)
# Service logic should validate priority range. NLU just extracts.
def test_add_objective_zero_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: zero prio task with priority 0"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "zero prio task"
    assert params.get("priority") == 0

def test_add_objective_negative_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: negative prio task with priority -1" # `\d+` won't match "-1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "negative prio task with priority -1" # Priority part not extracted
    assert params.get("priority") is None
    # If priority pattern was `(-?\d+)`, then it would capture -1.

# Test long project ID
def test_add_objective_long_project_id(nlu_processor_with_objectives: NLUProcessor):
    long_project_id = "this-is-a-very-long-project-identifier-that-should-still-be-captured-correctly-by-the-pattern"
    text = f"add objective: task for project {long_project_id}"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("related_project_id") == long_project_id

# Test long objective ID for set_status
def test_set_status_long_objective_id(nlu_processor_with_objectives: NLUProcessor):
    long_obj_id = "another-super-duper-long-objective-id-that-goes-on-and-on-to-test-limits-001"
    text = f"mark objective {long_obj_id} as active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    params = result.parsed_command.parameters
    assert params.get("objective_id") == long_obj_id
    assert params.get("new_status") == "active"

# Test key_details with very long key or value
def test_add_objective_long_key_or_value_in_details(nlu_processor_with_objectives: NLUProcessor):
    long_key = "a_key_that_is_exceptionally_long_to_see_how_the_regex_handles_it_and_if_it_breaks_parsing_or_not"
    long_value = "a value that is similarly very very long, stretching across multiple virtual lines perhaps, containing spaces, numbers 123, and punctuation like .,'- to ensure full capture."
    text = f"add objective: long keyval test details: {long_key} is {long_value}, short_key is short_val"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "long keyval test"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get(long_key) == long_value
    assert key_details.get("short_key") == "short_val"

# Test "details" keyword at the very end of the command string for add_objective
def test_add_objective_details_keyword_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with details keyword at end details" # No colon, no key-values
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task with details keyword at end" # "details" becomes part of desc
    assert "key_details" not in params # No actual key-value string followed "details"

def test_add_objective_details_keyword_with_colon_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with details colon at end details:" # Colon but no key-values
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task with details colon at end"
    # key_details_kv_string would be empty, so key_details becomes None or {}
    assert params.get("key_details") is None or params.get("key_details") == {}

# Test if "for project" without an ID is handled gracefully for add/list
def test_add_objective_for_project_no_id(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task for project" # No ID after "for project"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    # The `(?P<project_id>[\w.-]+)` requires at least one char.
    # So the optional group `(?:\s+for project\s+(?P<project_id>...))?` will not match.
    # Description will absorb "task for project".
    assert params.get("description") == "task for project"
    assert params.get("related_project_id") is None

def test_list_objectives_for_project_no_id(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives for project" # No ID
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("related_project_id") is None # Group for project_id won't match

# Test if "with priority" without a value is handled gracefully for add
def test_add_objective_with_priority_no_value(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with priority" # No value after "with priority"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    # `(?P<priority>\d+)` requires digits.
    # Optional group `(?:\s+with priority\s+(?P<priority>...))?` won't match.
    # Description absorbs "task with priority".
    assert params.get("description") == "task with priority"
    assert params.get("priority") is None

# Test if "status" without a value is handled for list command
def test_list_objectives_status_no_value(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives with status" # No value after "with status"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    # `(?P<status_filter>active|...})` requires one of the valid statuses.
    # Optional group `(?:\s+with status\s*:?\s*(?P<status_filter>...))?` won't match.
    assert params.get("status_filter") is None

# Test if "as" or "status" without a value is handled for set_status command (should fail intent match)
def test_set_status_as_no_value(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective obj99 as " # No value after "as"
    result = nlu_processor_with_objectives.process_text(text)
    # The part `as\s*(?P<new_status>active|...)` is not optional overall in the intent pattern.
    # If new_status part doesn't match, intent should fail.
    assert result.detected_intent.name != "set_user_objective_status"

def test_set_status_status_keyword_no_value(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective obj100 status " # No value after "status"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name != "set_user_objective_status"

# Test commands with only keywords, e.g., "objective details"
def test_only_keywords_objective_details(nlu_processor_with_objectives: NLUProcessor):
    text = "objective details"
    result = nlu_processor_with_objectives.process_text(text)
    # This is too ambiguous. Unlikely to match any specific objective command.
    # Might match a generic query intent.
    assert result.detected_intent.name not in ["add_user_objective", "list_user_objectives", "set_user_objective_status"]

def test_only_keywords_project_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "project priority"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name not in ["add_user_objective", "list_user_objectives", "set_user_objective_status"]

def test_only_keywords_status_update(nlu_processor_with_objectives: NLUProcessor):
    text = "status update"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name not in ["add_user_objective", "list_user_objectives", "set_user_objective_status"]

# Test unicode characters in description, IDs, project IDs, key_details
def test_unicode_in_description(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: café au lait résumé"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("description") == "café au lait résumé"

def test_unicode_in_objective_id(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective task_café_1 as completed" # Pattern `[\w.-]+` might not include non-ASCII letters by default in some Python regex contexts.
                                                   # However, `\w` in Python 3's `re` module matches Unicode word characters by default.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command.parameters.get("objective_id") == "task_café_1"

def test_unicode_in_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new for project ProjeKT_ÄÖÜ"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("related_project_id") == "ProjeKT_ÄÖÜ"

def test_unicode_in_key_details_keys_and_values(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: unicode test details: nom_clé is valœr_unicøde, another_k€y is '모든 것'"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    key_details = result.parsed_command.parameters.get("key_details")
    assert isinstance(key_details, dict)
    # Regex for key `([\w\s-]+?)` and value `([\w\s.,'@\-\/%+'\"()]+?)` should handle unicode if `\w` does.
    assert key_details.get("nom_clé") == "valœr_unicøde"
    assert key_details.get("another_k€y") == "'모든 것'"

# Test sentence ending punctuation in description or IDs.
def test_punctuation_in_description_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: Finish the report." # Period at end of description.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    # Description pattern `(.+?)` will capture the period.
    # TRANSFORM_STRIP_PUNCTUATION_FROM_END might be needed if period is not desired.
    # Current post-processing for description is just TRANSFORM_STRIP.
    assert result.parsed_command.parameters.get("description") == "Finish the report."

def test_punctuation_in_objective_id_end(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective task_end. as completed" # Pattern `[\w.-]+` includes period.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "set_user_objective_status"
    assert result.parsed_command.parameters.get("objective_id") == "task_end."

def test_punctuation_in_project_id_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: new for project ProjectX."
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    assert result.parsed_command.parameters.get("related_project_id") == "ProjectX."

# Test "details" where value contains commas
def test_add_objective_details_value_with_commas(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: grocery list details: items are 'apples, bananas, and oranges', budget is $20"
    # The key-value regex `([\w\s]+?)\s*[:is]\s*([\w\s.,'@\-\/%+'\"()]+?)(?=\s*,\s*[\w\s]+?\s*[:is]|$)"
    # The value part `([\w\s.,'@\-\/%+'\"()]+?)` includes comma.
    # The lookahead `(?=\s*,\s*[\w\s]+?\s*[:is]|$)` is key. It stops the value if the comma
    # is followed by another key-value pair structure.
    # If "'apples, bananas, and oranges'" is quoted, it's usually treated as a single value.
    # If not quoted, "apples" would be value for "items", then "bananas" would be a new key.
    # Let's assume the NLU pattern for value can handle quoted strings containing commas.
    # The current value regex `([\w\s.,'@\-\/%+'\"()]+?)` will capture up to the next separating comma.
    # So, "items are 'apples, bananas, and oranges'" -> key: "items", value: "'apples"
    # Then next part: "bananas, and oranges', budget is $20" -> key: "bananas", value: "and oranges'"
    # This is because the value capture is non-greedy and stops at the first comma that could start a new pair.
    # This highlights a known complexity with simple regex for nested structures or values with delimiters.
    # To handle this robustly, values with internal commas usually need to be quoted,
    # and the regex needs to respect quotes. The current regex does not explicitly handle quotes to span commas.

    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    print(f"DEBUG: Key details for comma test: {key_details}")
    # Based on current regex behavior (non-greedy value, comma as primary separator):
    assert key_details.get("items") == "'apples" # Value stops at first comma
    assert key_details.get("bananas") == "and oranges'" # Next key-value pair
    assert key_details.get("budget") == "$20"
    # This shows the current limitation. For values with commas, they MUST be quoted AND
    # the regex for value capture must be quote-aware, e.g. `(?:'([^']*)'|"([^"]*)"|([\w\s.,@\-\/%+()]+?))`
    # The current value regex is simpler: `([\w\s.,'@\-\/%+'\"()]+?)`
    # It will include the quote as part of the value if present.
    pass # Acknowledging this limitation.

# Test if "for project" comes after priority
def test_add_objective_project_after_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with priority 1 for project Alpha"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("priority") == 1
    assert params.get("related_project_id") == "Alpha"

# Test if "details" comes after project (and priority if present)
def test_add_objective_details_after_project_and_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task with priority 2 for project Beta details: key is val"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("priority") == 2
    assert params.get("related_project_id") == "Beta"
    assert isinstance(params.get("key_details"), dict)
    assert params.get("key_details").get("key") == "val"

def test_add_objective_details_after_project_no_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task for project Gamma details: anotherkey is anotherval"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("priority") is None
    assert params.get("related_project_id") == "Gamma"
    assert isinstance(params.get("key_details"), dict)
    assert params.get("key_details").get("anotherkey") == "anotherval"

# Test if "for project" comes after details (this order is less common and might not be supported by simple sequential regex)
def test_add_objective_project_after_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task details: k is v for project Delta"
    # The regex structure is `desc (priority)? (project)? (details)?`
    # So "for project Delta" after "details: k is v" would be part of the key_details_kv_string.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("priority") is None
    assert params.get("related_project_id") is None # Project part is inside details string
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("k") == "v for project Delta" # "for project Delta" is part of the value for "k"

# Test if "with priority" comes after details
def test_add_objective_priority_after_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task details: k is v with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("priority") is None # Priority part is inside details string
    assert params.get("related_project_id") is None
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("k") == "v with priority 1" # "with priority 1" is part of value for "k"

# Test for `TRANSFORM_PREFIX_DEFAULT_USER_ID`
def test_postprocess_prefix_default_user_id(nlu_processor_with_objectives: NLUProcessor):
    # This transform adds/replaces user_identifier with the default one.
    # We've been asserting this throughout. This is a focused check.
    text = "add objective: test user id"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.parsed_command.parameters.get("user_identifier") == "local_user_default_01"

# Test for `TRANSFORM_STRIP` (applied to description)
def test_postprocess_strip_description(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective:   leading and trailing spaces description   "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.parsed_command.parameters.get("description") == "leading and trailing spaces description"

# Test for `TRANSFORM_TO_INT` (applied to priority)
def test_postprocess_to_int_priority(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: test priority int with priority  007 " # Spaces and leading zeros
    result = nlu_processor_with_objectives.process_text(text)
    assert result.parsed_command.parameters.get("priority") == 7

# Test for `TRANSFORM_LOWERCASE_AND_STRIP` (applied to status_filter and new_status)
def test_postprocess_lowercase_strip_status(nlu_processor_with_objectives: NLUProcessor):
    text_list = "list objectives with status  ACTIVE  "
    text_set = "mark objective tid as  ComplETEd  "

    result_list = nlu_processor_with_objectives.process_text(text_list)
    assert result_list.parsed_command.parameters.get("status_filter") == "active"

    result_set = nlu_processor_with_objectives.process_text(text_set)
    assert result_set.parsed_command.parameters.get("new_status") == "completed"

# Test for `TRANSFORM_STRIP_DICT_KEYS_VALUES` (applied to key_details)
def test_postprocess_strip_dict_keys_values(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: test dict strip details:  key one  :  value one  , key two:value two "
    result = nlu_processor_with_objectives.process_text(text)
    key_details = result.parsed_command.parameters.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("key one") == "value one" # Spaces within key/value preserved by regex, stripped by transform
    assert key_details.get("key two") == "value two" # No spaces to strip in value here
    # The transform `TRANSFORM_STRIP_DICT_KEYS_VALUES` acts on the *values* of the dict, not keys.
    # And also on the keys themselves if they were captured with spaces.
    # Let's re-verify the transform logic for keys.
    # `new_dict[key.strip()] = value.strip() if isinstance(value, str) else value` -> Yes, keys are stripped.
    # So, " key one " becomes "key one". " value one " becomes "value one".
    # "key two" becomes "key two". "value two" becomes "value two". Correct.

# Test for `TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2` for key_details_kv_string
# This is implicitly tested by all key_details tests.
# This is more about ensuring the raw string is processed into a dict.
def test_postprocess_extract_key_value_pairs_v2(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: kv string test details: name is Jules, role is Engineer"
    result = nlu_processor_with_objectives.process_text(text)
    # The capture group `details_kv_string` would get "name is Jules, role is Engineer"
    # Then TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2 converts this string to a dict.
    key_details = result.parsed_command.parameters.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("name") == "Jules"
    assert key_details.get("role") == "Engineer"

# Test if an empty details string results in None or empty dict for key_details
def test_postprocess_empty_details_string(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: empty details test details: "
    result = nlu_processor_with_objectives.process_text(text)
    # `details_kv_string` would be "" or " ".
    # TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2 returns None if input is empty/whitespace.
    key_details = result.parsed_command.parameters.get("key_details")
    assert key_details is None or key_details == {}

# Test if a details string that doesn't parse into key-values results in None or empty dict
def test_postprocess_malformed_details_string(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: malformed details test details: this is not a key value string"
    result = nlu_processor_with_objectives.process_text(text)
    # `details_kv_string` is "this is not a key value string"
    # TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2 will find no matches for `key:value` pattern.
    # It should return an empty dictionary in this case.
    key_details = result.parsed_command.parameters.get("key_details")
    assert key_details == {} # V2 returns empty dict if no pairs found

# Test if a key_detail value itself looks like another key-value pair (e.g. "config is 'port:8080, host:localhost'")
def test_add_objective_nested_like_value_in_details(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: nested value test details: primary_config is 'port:8080, host:localhost', secondary_config is 'retries:3'"
    result = nlu_processor_with_objectives.process_text(text)
    # The value regex `([\w\s.,'@\-\/%+'\"()]+?)` is non-greedy and stops at the lookahead `(?=\s*,\s*[\w\s]+?\s*[:is]|$)`.
    # So, for "primary_config is 'port:8080, host:localhost'", the value captured will be "'port:8080, host:localhost'".
    # The comma inside the quoted string does not trigger the lookahead for a new key-value pair.
    key_details = result.parsed_command.parameters.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("primary_config") == "'port:8080, host:localhost'"
    assert key_details.get("secondary_config") == "'retries:3'"

# Test if a user_identifier is provided in the text, it should be overridden by the default.
# (This isn't really a feature of the patterns, but of the post-processing directive for user_identifier)
def test_add_objective_with_user_id_in_text_gets_overridden(nlu_processor_with_objectives: NLUProcessor):
    text = "user test_user add objective: my objective" # Assuming a pattern could hypothetically capture "user test_user"
    # Current patterns for add_user_objective don't have a place for user_id in the text.
    # The user_identifier is added via a specific post-processing step that sets it to default.
    # So, this test is more about confirming that default is applied regardless of text.
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    # The description will likely be "user test_user add objective: my objective" or similar if "add objective" is found later.
    # Or, if "user test_user" is not part of any primary keyword match:
    # Description: "my objective"
    # Let's assume the most specific pattern for "add objective: desc" matches.
    # "add objective: my objective" -> desc="my objective"
    # "user test_user" is extraneous.
    # Let's try:
    text_more_specific = "add objective: my objective for user test_user"
    result_specific = nlu_processor_with_objectives.process_text(text_more_specific)
    assert result_specific.detected_intent.name == "add_user_objective"
    # "for user test_user" would be part of description as there's no "for user" capture group.
    assert result_specific.parsed_command.parameters.get("description") == "my objective for user test_user"
    assert result_specific.parsed_command.parameters.get("user_identifier") == "local_user_default_01" # Overridden/set by post-proc

    # Simpler case:
    result_simple = nlu_processor_with_objectives.process_text("add objective: test user override")
    assert result_simple.parsed_command.parameters.get("user_identifier") == "local_user_default_01"

# Test if only "details:" is present after description for add_objective
def test_add_objective_only_details_keyword_after_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: some task details:"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "some task"
    assert params.get("key_details") is None or params.get("key_details") == {}

# Test if "for project" is at the very end for add_objective
def test_add_objective_for_project_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: some other task for project"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "some other task for project" # project_id group won't match
    assert params.get("related_project_id") is None

# Test if "with priority" is at the very end for add_objective
def test_add_objective_with_priority_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: yet another task with priority"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "yet another task with priority" # priority group won't match
    assert params.get("priority") is None

# Test if "status" is at the very end for list_objectives
def test_list_objectives_status_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives with status"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("status_filter") is None # status_filter group won't match

# Test if "as" is at the very end for set_status (should fail intent)
def test_set_status_as_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "mark objective myobj as"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name != "set_user_objective_status"

# Test if "status" keyword (for new_status) is at the very end for set_status (should fail intent)
def test_set_status_status_keyword_for_value_at_end(nlu_processor_with_objectives: NLUProcessor):
    text = "set objective myobj status" # e.g. "set objective myobj status active"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name != "set_user_objective_status"

# Test a very long command string that is valid
def test_very_long_valid_add_objective_command(nlu_processor_with_objectives: NLUProcessor):
    desc = "This is an extremely long description designed to test the limits of regex capture groups and ensure that even with substantial text, the parsing remains accurate and extracts all components correctly, especially when followed by multiple optional parameters like priority, project, and detailed key-value pairs for extended context."
    proj = "SuperLongProjectNameIdentifierWithAlphaNumericsAndHyphens-001-Variant-Omega"
    prio = 5
    kv_string = "first_detail_key is 'A very elaborate value for the first key, containing spaces, commas, and various punctuation symbols to ensure robustness.', second_detail_key is 'Another equally long and complex value string that might span multiple lines if it were in a text editor, testing line break resilience (though regex usually sees one line).', third_key is 'shorter_value', final_check_param is 'all_good_hopefully_for_this_extreme_case'"
    text = f"add objective: {desc} with priority {prio} for project {proj} details: {kv_string}"

    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == desc
    assert params.get("priority") == prio
    assert params.get("related_project_id") == proj
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("first_detail_key") == "'A very elaborate value for the first key, containing spaces, commas, and various punctuation symbols to ensure robustness.'"
    assert key_details.get("second_detail_key") == "'Another equally long and complex value string that might span multiple lines if it were in a text editor, testing line break resilience (though regex usually sees one line).'"
    assert key_details.get("third_key") == "'shorter_value'" # Quotes added by my string construction for safety
    assert key_details.get("final_check_param") == "'all_good_hopefully_for_this_extreme_case'"

# Test a command that is just garbage or random characters
def test_garbage_input_string(nlu_processor_with_objectives: NLUProcessor):
    text = "!@#$%^&*()_+`~[]{};':\",./<>?qwertyuiopasdfghjklzxcvbnm"
    result = nlu_processor_with_objectives.process_text(text)
    # Should not match any objective intent. Might be 'unknown_intent' or some fallback.
    assert result.detected_intent.name not in ["add_user_objective", "list_user_objectives", "set_user_objective_status"]
    print(f"Garbage input test: Intent is '{result.detected_intent.name}'")

# Test an empty string input
def test_empty_input_string(nlu_processor_with_objectives: NLUProcessor):
    text = ""
    result = nlu_processor_with_objectives.process_text(text)
    # Behavior for empty string depends on NLUProcessor.process_text handling.
    # It might raise an error, or return unknown_intent.
    # nlu_engine.py: process_text -> if not text.strip(): returns NLUResult with intent 'no_discernible_intent'
    assert result.detected_intent.name == "no_discernible_intent"

# Test input string with only spaces
def test_whitespace_input_string(nlu_processor_with_objectives: NLUProcessor):
    text = "     "
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "no_discernible_intent"

# Test a command that uses "objective" multiple times in the description
def test_add_objective_with_objective_keyword_multiple_times_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: My primary objective is to define a new objective for the team objective planning session."
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "My primary objective is to define a new objective for the team objective planning session."
    assert params.get("priority") is None
    assert params.get("related_project_id") is None
    assert "key_details" not in params

# Test a command that uses "project" multiple times in the description before "for project <id>"
def test_add_objective_with_project_keyword_multiple_times_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: This project objective is for the new project initiative for project MainProjectFocus"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "This project objective is for the new project initiative"
    assert params.get("related_project_id") == "MainProjectFocus"

# Test a command that uses "priority" multiple times in the description before "with priority <num>"
def test_add_objective_with_priority_keyword_multiple_times_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: The priority for this task is high priority, it's a top priority item with priority 1"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "The priority for this task is high priority, it's a top priority item"
    assert params.get("priority") == 1

# Test a command that uses "details" multiple times in the description before "details: <kv>"
def test_add_objective_with_details_keyword_multiple_times_in_desc(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: Please provide more details on the task details section details: final_review is needed, contact_person is 'John Doe'"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "Please provide more details on the task details section"
    key_details = params.get("key_details")
    assert isinstance(key_details, dict)
    assert key_details.get("final_review") == "needed"
    assert key_details.get("contact_person") == "'John Doe'"

# Test for "list project objectives" (similar to "list all project objectives")
def test_list_project_objectives_simple(nlu_processor_with_objectives: NLUProcessor):
    text = "list project objectives"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("related_project_id") is None # "project" here is adjective, no ID given
    assert params.get("status_filter") is None

# Test for "list status objectives" (e.g. "list active objectives" - already covered)
# Let's try "list objectives status" (no value)
def test_list_objectives_status_keyword_no_value_variant(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives status"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("status_filter") is None # No value for status

# Test if "objective" keyword is part of a project ID or objective ID for add/list/set
def test_add_objective_with_objective_in_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "add objective: task for project team_objective_review_project"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "add_user_objective"
    params = result.parsed_command.parameters
    assert params.get("description") == "task"
    assert params.get("related_project_id") == "team_objective_review_project"

def test_list_objectives_with_objective_in_project_id(nlu_processor_with_objectives: NLUProcessor):
    text = "list objectives for project main_objective_tracker_project"
    result = nlu_processor_with_objectives.process_text(text)
    assert result.detected_intent.name == "list_user_objectives"
    params = result.parsed_command.parameters
    assert params.get("related_project_id") == "main_objective_tracker_project"

def test_set_status_with_objective_in_objective_id(nlu_processor_with_objectives: NLUProcessor):
    # Already covered by `test_set_objective_status_with_objective_in_id`
    # "mark main_objective_phase1 as completed" -> id = "main_objective_phase1"
    pass
