import pytest
from src.core.prompt_constructors import construct_main_planning_prompt

# Minimal mock inputs for the prompt constructor
MOCK_USER_PROMPT = "Test user prompt"
MOCK_HISTORY_CONTEXT = "User: Hello\nAssistant: Hi there!"
MOCK_NLU_INFO = "NLU Analysis: Intent=Test, Entities=[]"
MOCK_GENERAL_KB_CONTEXT = "General KB: Info about X."
MOCK_KG_DERIVED_CONTEXT = "KG Derived: Item A related to Topic B."
MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT = "Past Plan Summary: Plan P1 succeeded."
MOCK_PLAN_LOG_INSIGHTS = "Plan Log Insights: Step S1 often fails."
MOCK_FEEDBACK_INSIGHTS_CONTEXT = "Feedback Insights: Users like feature F."
MOCK_AGENT_DESC = "- AgentZ: Does Z."

class TestConstructMainPlanningPromptStrategies:

    def test_strategy_default(self):
        prompt = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC,
            planner_strategy="Strategy_Default" # Explicitly test default
        )
        # Default should not contain unique phrases from other strategies
        assert "FOCUS ON CLARITY" not in prompt.upper()
        assert "CLARITY and EXPLICITNESS" not in prompt.upper()
        assert "PRIORITIZE BREVITY" not in prompt.upper()
        assert "BREVITY and EFFICIENCY" not in prompt.upper()
        # It should contain the standard context usage instructions
        assert "When creating the plan, consider the following:" in prompt
        assert "TASK: Based on ALL the above information" in prompt # Check standard task intro

    def test_strategy_default_when_none_or_unspecified(self):
        prompt_none = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC,
            planner_strategy=None # Test None explicitly
        )
        prompt_unspecified = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC
            # planner_strategy defaults to "Strategy_Default"
        )
        for prompt in [prompt_none, prompt_unspecified]:
            assert "FOCUS ON CLARITY" not in prompt.upper()
            assert "CLARITY and EXPLICITNESS" not in prompt.upper()
            assert "PRIORITIZE BREVITY" not in prompt.upper()
            assert "BREVITY and EFFICIENCY" not in prompt.upper()
            assert "When creating the plan, consider the following:" in prompt

    def test_strategy_focus_clarity(self):
        prompt = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC,
            planner_strategy="Strategy_FocusClarity"
        )
        # Check for clarity-specific phrases (case-insensitive for robustness)
        assert "CURRENT PLANNING STRATEGY: FOCUS ON CLARITY" in prompt
        assert "CLARITY and EXPLICITNESS" in prompt # From designed instructions
        assert "unambiguous purpose" in prompt
        # Check that brevity phrases are NOT present
        assert "PRIORITIZE BREVITY" not in prompt.upper()
        assert "BREVITY and EFFICIENCY" not in prompt.upper()
        # Ensure the main task instruction is still there and mentions strategy
        assert "TASK: Based on ALL the above information (especially noting the CURRENT PLANNING STRATEGY if specified)" in prompt

    def test_strategy_prioritize_brevity(self):
        prompt = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC,
            planner_strategy="Strategy_PrioritizeBrevity"
        )
        # Check for brevity-specific phrases
        assert "CURRENT PLANNING STRATEGY: PRIORITIZE BREVITY" in prompt
        assert "BREVITY and EFFICIENCY" in prompt # From designed instructions
        assert "minimum number of steps" in prompt
        # Check that clarity phrases are NOT present
        assert "FOCUS ON CLARITY" not in prompt.upper()
        assert "CLARITY and EXPLICITNESS" not in prompt.upper()
        # Ensure the main task instruction is still there
        assert "TASK: Based on ALL the above information (especially noting the CURRENT PLANNING STRATEGY if specified)" in prompt

    def test_unknown_strategy_falls_back_to_default(self):
        prompt = construct_main_planning_prompt(
            MOCK_USER_PROMPT, MOCK_HISTORY_CONTEXT, MOCK_NLU_INFO,
            MOCK_GENERAL_KB_CONTEXT, MOCK_KG_DERIVED_CONTEXT, MOCK_KG_PAST_PLAN_SUMMARY_CONTEXT,
            MOCK_PLAN_LOG_INSIGHTS, MOCK_FEEDBACK_INSIGHTS_CONTEXT, MOCK_AGENT_DESC,
            planner_strategy="Strategy_UnknownAndNonExistent" # An undefined strategy
        )
        # Should behave like default: no specific additional instructions for clarity or brevity
        assert "FOCUS ON CLARITY" not in prompt.upper()
        assert "PRIORITIZE BREVITY" not in prompt.upper()
        assert "When creating the plan, consider the following:" in prompt
        # The specific "CURRENT PLANNING STRATEGY" header for unknown strategies is not added, which is fine.
        # The main task instruction should be the one that mentions strategy generally.
        assert "TASK: Based on ALL the above information (especially noting the CURRENT PLANNING STRATEGY if specified)" in prompt
```
