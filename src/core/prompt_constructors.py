import json
from typing import List, Dict, Optional

# The function get_relevant_history_for_prompt is now removed.
# Conversation history processing is handled by ConversationContextManager in TerminusOrchestrator
# and the formatted history string is passed directly to the prompt constructors.

def construct_kb_query_generation_prompt(user_prompt:str, history_context_string:str, nlu_info:str) -> str:
   """
   Constructs a prompt for an LLM to generate a Knowledge Base search query.
   Args:
       user_prompt: The current user's request.
       history_context_string: A pre-formatted string of relevant conversation history.
       nlu_info: NLU analysis output for the current user_prompt.
   Returns:
       A string prompt for the LLM.
   """
   return ( f"Conversation History:\n{history_context_string}\n\n" # Use the passed string
            f"Current User Request: '{user_prompt}'\n"
            f"NLU Analysis of Request (Primary Intent, Entities, Implicit Goals, Alternatives):\n{nlu_info}\n\n" # Enhanced nlu_info
            f"Your task: Based on request, history, and NLU, generate a concise search query (max 5-7 words) for a knowledge base (KB) containing general info (docs, web content, code explanations). "
            f"**Strongly consider NLU entities and implicit goals for a targeted query.** " # Added implicit goals
            f"If no KB query seems useful, output ONLY: NO_QUERY_NEEDED\n"
            f"Otherwise, output ONLY the query string.\nSearch Query or Marker:" )

# Was: _format_kb_entry_for_prompt(self, kb_hit:Dict) -> str
def format_kb_entry_for_prompt(kb_hit:Dict) -> str:
   doc_preview = kb_hit.get('document','N/A')[:250]+"..."
   meta = kb_hit.get('metadata',{})
   meta_parts = [f"{k}: {str(v)[:50]}" for k,v in meta.items() if k not in ['document_content_ vezi', 'extracted_keywords', 'extracted_topics']]
   meta_preview = "; ".join(meta_parts)[:150]+"..."
   kw_preview = f" (Keywords: {str(meta.get('extracted_keywords','N/A'))[:70]}...)" if meta.get("extracted_keywords") else ""
   tpc_preview = f" (Topics: {str(meta.get('extracted_topics','N/A'))[:70]}...)" if meta.get("extracted_topics") else ""
   return f"  - Content Preview: \"{doc_preview}\" (Metadata: {meta_preview}){kw_preview}{tpc_preview}"

# Was: _format_plan_log_entry_for_prompt(self, kb_hit:Dict) -> str
def format_plan_log_entry_for_prompt(kb_hit:Dict) -> str:
   try:
       log_doc_str = kb_hit.get("document")
       if not log_doc_str: return "- Malformed plan log (missing document)."
       log_data = json.loads(log_doc_str)
       status = log_data.get("execution_summary",{}).get("overall_status","N/A")
       req_preview = log_data.get("original_user_request","N/A")[:75]+"..."
       user_sum = log_data.get("user_facing_plan_outcome_summary","N/A")[:100]+"..."
       intent = log_data.get("nlu_analysis_on_request",{}).get("intent","N/A")
       log_meta_kws = kb_hit.get("metadata",{}).get("extracted_keywords")
       log_meta_topics = kb_hit.get("metadata",{}).get("extracted_topics")
       kws_str = f" (Log Keywords: {str(log_meta_kws)[:70]}...)" if log_meta_kws else ""
       topics_str = f" (Log Topics: {str(log_meta_topics)[:70]}...)" if log_meta_topics else ""
       return (f"- Past plan for intent '{intent}' (request: '{req_preview}') -> Status: '{status}'. "
               f"User summary: '{user_sum}'.{kws_str}{topics_str}")
   except Exception as e:
       # print(f"Warning: Error formatting plan log for prompt: {e}") # Consider logging if this module has a logger
       return f"- Error processing plan log: {str(e)[:100]}"

# Was: _format_feedback_report_for_prompt(self, kb_hit:Dict) -> str
def format_feedback_report_for_prompt(kb_hit:Dict) -> str:
    try:
        report_doc_str = kb_hit.get("document")
        if not report_doc_str: return "- Malformed feedback report (missing document)."
        report_data = json.loads(report_doc_str)
        sent_dist = report_data.get("overall_sentiment_distribution", {})
        pos_perc = sent_dist.get('positive', 0.0) * 100
        neg_perc = sent_dist.get('negative', 0.0) * 100
        insights_preview = str(report_data.get("actionable_insights", report_data.get("comment_previews", []))[:1])[:150]
        return (f"Feedback Insights: Overall Sentiment (Pos: {pos_perc:.1f}%, Neg: {neg_perc:.1f}%). "
                f"Sample Insight/Comment: '{insights_preview}...'.")
    except Exception as e:
        # print(f"Warning: Error formatting feedback report for prompt: {e}")
        return f"- Error processing feedback report: {str(e)[:100]}"

# Was: _construct_main_planning_prompt(self, user_prompt:str, history_context:str, nlu_info:str, general_kb_context:str, plan_log_insights:str, feedback_insights_context:str, agent_desc:str) -> str
def construct_main_planning_prompt(user_prompt:str, history_context:str, nlu_info:str,
                                 general_kb_context:str, kg_derived_context:str,
                                 kg_past_plan_summary_context:str,
                                 plan_log_insights:str, feedback_insights_context:str,
                                 agent_desc:str,
                                 planner_strategy: Optional[str] = "Strategy_Default",
                                 active_objectives_string: Optional[str] = None) -> str:
   """
   Constructs the main, detailed prompt for the MasterPlanner LLM to generate an execution plan.

   Args:
       user_prompt: The current user's request.
       history_context: Formatted string of relevant conversation history.
       nlu_info: Formatted string of NLU analysis for the current request.
       general_kb_context: Context retrieved from ChromaDB based on semantic search.
       kg_derived_context: Context retrieved from Knowledge Graph based on entity/topic links.
       kg_past_plan_summary_context: Summaries of past simplified plans from KG for similar intents.
       plan_log_insights: Formatted insights from past detailed plan execution logs (ChromaDB).
       feedback_insights_context: Formatted insights from user feedback analysis reports (ChromaDB).
       agent_desc: Description of available agents and their capabilities.
       planner_strategy: The conceptual strategy to guide LLM's planning approach.
                         Expected values: "Strategy_Default", "Strategy_FocusClarity",
                                          "Strategy_PrioritizeBrevity".
                         Modifies instructional text within the prompt.

   Returns:
       A string prompt for the MasterPlanner LLM.
   """
   kb_section = ""
   if general_kb_context.strip(): kb_section += general_kb_context
   if kg_derived_context.strip(): kb_section += kg_derived_context
   if kg_past_plan_summary_context.strip(): kb_section += kg_past_plan_summary_context
   if plan_log_insights.strip(): kb_section += plan_log_insights
   if feedback_insights_context.strip(): kb_section += feedback_insights_context

   strategy_specific_instructions = ""
   if planner_strategy == "Strategy_FocusClarity":
       strategy_specific_instructions = (
           "\n--- CURRENT PLANNING STRATEGY: FOCUS ON CLARITY ---\n"
           "Your primary goal for this plan is CLARITY and EXPLICITNESS. Each step must be clearly defined with an unambiguous purpose. "
           "If intermediate data transformations are needed, define them as separate steps. Prefer descriptive task prompts for agents. "
           "Ensure dependencies are explicitly clear. Avoid overly complex or deeply nested plan structures if a flatter, "
           "clearer alternative exists, even if it means a few more simple steps.\n"
       )
   elif planner_strategy == "Strategy_PrioritizeBrevity":
       # Apply Brevity-focused instructions
       strategy_specific_instructions = (
           "\n--- CURRENT PLANNING STRATEGY: PRIORITIZE BREVITY ---\n"
           "Your primary goal for this plan is BREVITY and EFFICIENCY. Aim for the minimum number of steps required to achieve the user's core request. "
           "Prefer direct approaches and simpler agent interactions. Task prompts for agents should be concise. Only include essential dependencies.\n"
       )
   # If planner_strategy is "Strategy_Default" or unknown, strategy_specific_instructions remains empty,
   # and the LLM relies on the standard context_usage_instructions and overall task.

   objectives_section = ""
   if active_objectives_string and active_objectives_string.strip() and "No specific long-term objectives" not in active_objectives_string :
       objectives_section = (
           f"\n--- ACTIVE LONG-TERM USER OBJECTIVES ---\n"
           f"{active_objectives_string}\n"
           f"(These are overarching goals. Consider how the current request might align with or contribute to them.)\n"
       )

   context_usage_instructions = (
       "When creating the plan, consider the following:\n"
       "1. The 'NLU Analysis' provides the primary intent, confidence score, any alternative intents, extracted entities, and potentially implicit user goals for the CURRENT request. Use all these NLU facets to deeply understand the user's needs.\n"
       "2. Context from various Knowledge Base sources is provided: 'General Context' (semantic search), 'Knowledge Graph Derived Context' (entity/topic links), 'Past Simplified Plan Structures' (for similar intents), 'Insights from Past Plan Executions' (detailed logs), and 'Feedback Insights'. Use all available context to learn from past successes, failures, and user feedback.\n"
       "3. If 'Extracted Keywords' or 'Extracted Topics' are listed with any KB items, these can help refine task prompts or agent choices.\n"
       "4. Review 'Past Simplified Plan Structures' for similar intents. Note their success/failure, agent sequences, and key entities to inform your current plan. Avoid repeating past failures if possible.\n"
       "5. Agent 'Complexity' (low, medium, high) and 'Speed' (fast, medium, slow) ratings should guide agent selection. \n"
       "   - For simple tasks, favor agents with low complexity.\n"
       "   - For complex tasks, you might need high complexity agents; consider if the task can be broken down, especially if NLU indicates multiple intents or complex implicit goals.\n"
       "6. The overall 'Request Priority' (e.g., high, normal, low - assume 'normal' if not specified) should influence your choices:\n" # Index updated
       "   - For 'high' priority requests, aim for quicker plans. This might mean choosing agents with 'fast' or 'medium' speed. Address the primary intent and key implicit goals directly.\n"
       "   - For 'low' priority requests, you can afford more thoroughness. Agents with 'slow' speed or higher 'complexity' can be used. Consider exploring alternative intents if NLU suggests them.\n"
       "   - For 'normal' priority, balance speed, complexity, and result quality. Address primary intent and important implicit goals; consider alternatives if primary confidence is low.\n"
       "7. You may also assign a 'priority' field ('high', 'normal', 'low') to individual steps in your generated plan if you discern differential importance among sub-tasks based on the overall request priority or dependencies.\n" # Index updated
   )

   return (f"You are the MasterPlanner. Your role is to decompose a complex user request into a sequence of tasks for specialized AI agents.\n\n"
           f"{history_context}"
           f"--- KNOWLEDGE BASE & NLU CONTEXT ---\n"
           f"Current User Request: '{user_prompt}'\n"
           f"NLU Analysis of Current Request (Intent, Entities, Implicit Goals, Alternatives):\n{nlu_info}\n\n"
           f"{objectives_section}" # Added objectives section
           f"{kb_section if kb_section.strip() else 'No specific information retrieved from Knowledge Base for this request.'}\n"
           f"{context_usage_instructions}"
           f"{strategy_specific_instructions}\n" # Added strategy specific instructions here
           f"--- AVAILABLE AGENTS & TASK ---\n"
           f"Available specialized agents and their capabilities are:\n{agent_desc}\n\n"
           f"TASK: Based on ALL the above information (especially noting the CURRENT PLANNING STRATEGY if specified), create a detailed, step-by-step JSON plan to fulfill the user's current request. \n" # Emphasize strategy
           f"Plan Schema:\n"
           f"  - 'step_id': (String) Unique ID (e.g., \"1\", \"2a\", \"cond_check_user_pref\").\n"
           f"  - 'agent_name': (String) Agent from list OR 'parallel_group' OR 'conditional' OR 'loop' (if using step_type).\n"
           f"  - 'step_type': (Optional, String) Default is agent execution. Can be 'conditional', 'parallel_group', 'loop', or 'agent_service_call'. If specified, 'agent_name' might be redundant or set to the step_type (e.g., agent_name: 'conditional').\n"
           f"  - 'description': (Optional, String) Human-readable description of the step's purpose.\n"
           f"  - 'task_prompt': (String) Specific prompt for the agent. (Not used if 'parallel_group', 'conditional', or 'loop' body is defined by 'loop_body_step_ids').\n"
           f"  - 'dependencies': (Optional, List[String]) IDs of prior steps this step depends on.\n"
           f"  - 'output_variable_name': (Optional, String) Variable name for step's output (e.g., 'user_preference_value').\n"
           f"  - 'priority': (Optional, String) 'high', 'normal', or 'low' for this step.\n"
           f"  - 'max_retries': (Integer, Optional, Default: 0).\n"
           f"  - 'retry_delay_seconds': (Integer, Optional, Default: 5).\n"
           f"  - 'retry_on_statuses': (List[String], Optional, Default: [\"error\"]).\n"
           f"For 'parallel_group': include 'sub_steps': [List of standard step objects]. Sub-steps MUST be input-independent. Group's 'output_variable_name' will be a dict of sub-step outputs.\n"
           f"For 'conditional': include 'condition': {{'source_step_id':'id', 'source_output_variable':'path.to.var', 'operator':'equals|etc.', 'value':'compare_val', 'value_type':'string|etc'}}, 'if_true_step_id':'id_true', 'if_false_step_id':'id_false' (optional). Ensure 'dependencies' includes 'condition.source_step_id'.\n"
           f"For 'loop' ('loop_type':'while'): include 'condition': {{...as conditional...}}, 'loop_body_step_ids':['id1', 'id2'], 'max_iterations':10 (optional). The loop continues while condition is true. Loop body steps (id1, id2) must be defined elsewhere in the main plan list.\n"
           f"For 'agent_service_call': include 'target_agent_name':'AgentName', 'service_name':'ServiceName', 'service_params':{{'key':'val'}}. Use for direct, structured calls to other agents if their services are known.\n"
           f"For 'SystemCapabilityManager': if task_prompt is 'SUGGEST_NEW_TOOL', include 'suggested_tool_description': (String) detailing the missing capability.\n"
           f"Example Agent Step: {{'step_id': '1', 'agent_name': 'WebCrawler', 'task_prompt': 'Search for X', 'output_variable_name': 'search_X'}}\n"
           f"IMPORTANT: Output ONLY the raw JSON plan as a list of step objects. If unplannable or request is too simple for a plan, return an empty JSON list []." )

# Was: _construct_revision_planning_prompt(self, user_prompt:str, history_context:str, nlu_info:str, failure_details:Dict, agent_desc:str) -> str
def construct_revision_planning_prompt(user_prompt:str, history_context:str, nlu_info:str,
                                     failure_details:Dict, agent_desc:str) -> str:
   failed_plan_str = failure_details.get("plan_that_failed_this_attempt","Original plan not available.")
   failed_step_def_str = json.dumps(failure_details.get("failed_step_definition"),indent=2) if failure_details.get("failed_step_definition") else "N/A"
   failed_exec_res_str = json.dumps(failure_details.get("failed_step_execution_result"),indent=2) if failure_details.get("failed_step_execution_result") else "N/A"
   prior_outputs_str = json.dumps(failure_details.get("step_outputs_before_failure"),indent=2) if failure_details.get("step_outputs_before_failure") else "N/A"

   failure_context_section = (
       f"--- DETAILED FAILURE CONTEXT ---\n"
       f"Plan that Failed This Attempt:\n```json\n{failed_plan_str}\n```\n"
       f"Failed Step/Group Definition:\n```json\n{failed_step_def_str}\n```\n"
       f"Last Execution Result of Failed Step/Group:\n```json\n{failed_exec_res_str}\n```\n"
       f"Available Data Outputs from Prior Successful Steps (at time of failure):\n```json\n{prior_outputs_str}\n```\n"
       f"--- END DETAILED FAILURE CONTEXT ---\n\n" )
   return (f"You are MasterPlanner. A previous plan attempt failed. Analyze failure and provide revised JSON plan.\n\n"
           f"{history_context}"
           f"Original User Request: '{user_prompt}' (Consider its original priority if specified in context).\n"
           f"NLU Analysis (from first attempt - Intent, Entities, Implicit Goals, Alternatives):\n{nlu_info}\n\n" # Enhanced nlu_info usage
           f"{failure_context_section}"
           f"Available Agents (Note Complexity & Speed Ratings):\n{agent_desc}\n\n" # Added Speed
           f"Revision Instructions:\n1. Analyze 'DETAILED FAILURE CONTEXT'.\n"
           f"2. Goal: revised JSON plan. Make MINIMAL TARGETED changes to 'Plan that Failed This Attempt'.\n"
           f"3. Prioritize fixing/replacing failed step. Adjust subsequent steps if dependencies change.\n"
           f"4. Ensure coherence with original request. Return COMPLETE VALID JSON plan (same overall schema, including for conditional, loop, parallel, and agent_service_call steps if used).\n"
           f"5. Consider agent complexity and if a conditional branch or loop was involved in the failure.\n\n"
           f"IMPORTANT: Output ONLY raw JSON. If unsalvageable, return []." )
