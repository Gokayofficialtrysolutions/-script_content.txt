import json
from typing import List, Dict, Optional

# Was: _get_relevant_history_for_prompt(self, user_prompt:str, full_history:bool=False) -> str
def get_relevant_history_for_prompt(conversation_history: List[Dict], max_history_items: int, user_prompt:str, full_history:bool=False) -> str:
   history_to_consider = conversation_history[:-1] # Exclude current user_prompt which is handled separately
   if full_history or not history_to_consider: # if full_history or no prior turns except current
       relevant_turns = history_to_consider[-max_history_items:]
   else: # Select a smaller window if not full_history and there's enough history
       relevant_turns = history_to_consider[-4:] # Default to last 4 turns for concise context

   history_list = []
   for t in relevant_turns:
       role = t.get('role', 'unknown').capitalize()
       content = str(t.get('content', ''))
       history_list.append(f"{role}: {content}")
   history_str = "\n".join(history_list)
   return f"Relevant Conversation History:\n{history_str}\n\n" if history_str else "No relevant conversation history found.\n\n"

# Was: _construct_kb_query_generation_prompt(self, user_prompt:str, history_context:str, nlu_info:str) -> str
def construct_kb_query_generation_prompt(user_prompt:str, history_context:str, nlu_info:str) -> str:
   return ( f"{history_context}"
            f"Current User Request: '{user_prompt}'\n"
            f"NLU Analysis of Request: {nlu_info}\n\n"
            f"Your task: Based on request, history, and NLU, generate a concise search query (max 5-7 words) for a knowledge base (KB) containing general info (docs, web content, code explanations). "
            f"**Strongly consider NLU entities for a targeted query.** "
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
                                 general_kb_context:str, plan_log_insights:str, feedback_insights_context:str,
                                 agent_desc:str) -> str:
   kb_section = ""
   if general_kb_context.strip(): kb_section += general_kb_context
   if plan_log_insights.strip(): kb_section += plan_log_insights
   if feedback_insights_context.strip(): kb_section += feedback_insights_context

   context_usage_instructions = (
       "When creating the plan, consider the following:\n"
       "1. The 'NLU Analysis' provides key entities and the primary intent of the user's CURRENT request.\n"
       "2. 'General Context from Knowledge Base', 'Insights from Past Plan Executions', and 'Feedback Insights' offer background. Learn from past successes, failures, and user feedback.\n"
       "3. If 'Extracted Keywords' or 'Extracted Topics' are listed with any KB items, these can help refine task prompts or agent choices.\n"
       "4. Agent 'Complexity' ratings (low, medium, high) should guide agent selection: favor lower complexity for simple tasks, and consider breaking down tasks requiring high complexity agents.\n"
       "5. If an overall 'Request Priority' (e.g., high, normal, low) is specified for the user's request, consider this in your planning. High priority tasks might prefer more direct or faster plans, while low priority tasks can afford more thoroughness if it doesn't block others. You may also assign a 'priority' field ('high', 'normal', 'low') to individual steps in your generated plan if you discern differential importance among sub-tasks based on the overall request priority.\n"
   )

   return (f"You are the MasterPlanner. Your role is to decompose a complex user request into a sequence of tasks for specialized AI agents.\n\n"
           f"{history_context}"
           f"--- KNOWLEDGE BASE & NLU CONTEXT ---\n"
           f"Current User Request: '{user_prompt}'\n"
           f"NLU Analysis of Current Request: {nlu_info}\n\n"
           f"{kb_section if kb_section.strip() else 'No specific information retrieved from Knowledge Base for this request.'}\n"
           f"{context_usage_instructions}\n"
           f"--- AVAILABLE AGENTS & TASK ---\n"
           f"Available specialized agents and their capabilities are:\n{agent_desc}\n\n"
           f"TASK: Based on ALL the above information, create a detailed, step-by-step JSON plan to fulfill the user's current request. \n"
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
           f"NLU Analysis (from first attempt): {nlu_info}\n\n"
           f"{failure_context_section}"
           f"Available Agents (Note Complexity Ratings):\n{agent_desc}\n\n"
           f"Revision Instructions:\n1. Analyze 'DETAILED FAILURE CONTEXT'.\n"
           f"2. Goal: revised JSON plan. Make MINIMAL TARGETED changes to 'Plan that Failed This Attempt'.\n"
           f"3. Prioritize fixing/replacing failed step. Adjust subsequent steps if dependencies change.\n"
           f"4. Ensure coherence with original request. Return COMPLETE VALID JSON plan (same overall schema, including for conditional, loop, parallel, and agent_service_call steps if used).\n"
           f"5. Consider agent complexity and if a conditional branch or loop was involved in the failure.\n\n"
           f"IMPORTANT: Output ONLY raw JSON. If unsalvageable, return []." )
