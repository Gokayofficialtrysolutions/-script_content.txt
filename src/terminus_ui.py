import streamlit as st,asyncio,json,os,subprocess,time,datetime,uuid # Ensure time, datetime, and uuid are imported
from pathlib import Path # Path is used for temp_video_path
import pandas as pd,plotly.express as px,plotly.graph_objects as go # Keep Plotly for potential future Data Analysis UI
from typing import Optional # For type hinting in display_feedback_widgets

# Assuming master_orchestrator.py (and thus its instances) are in $INSTALL_DIR/agents/
# and terminus_ui.py is in $INSTALL_DIR/
# The launch_terminus.py script should handle adding $INSTALL_DIR to sys.path or PYTHONPATH
# so that `from agents.master_orchestrator import ...` works.
try:
    from agents.master_orchestrator import orchestrator, doc_processor, web_intel
except ImportError as e:
    st.error(f"Critical Error: Could not import orchestrator components: {e}. UI cannot function.")
    st.stop() # Halt execution of the UI if core components can't be imported


st.set_page_config(page_title="TERMINUS AI",layout="wide",initial_sidebar_state="expanded")

# Helper function for feedback widgets
def display_feedback_widgets(item_id: str, item_type: str,
                             key_suffix: str, # To ensure unique widget keys
                             current_operation_mode: str,
                             related_prompt_preview: Optional[str] = None):
    # Ensure orchestrator is available (it should be if UI is running)
    if 'orchestrator' not in globals() or orchestrator is None:
        st.warning("Feedback system unavailable: Orchestrator not loaded.")
        return

    st.write("---") # Visual separator for feedback section
    st.caption("Rate this response/item:")

    # Use a more robust unique key for session state based on item_id and type
    feedback_submitted_key = f"feedback_submitted_{item_type}_{item_id}_{key_suffix}"
    comment_key = f"comment_{item_type}_{item_id}_{key_suffix}"

    if feedback_submitted_key not in st.session_state:
        st.session_state[feedback_submitted_key] = None # None, "Positive", "Negative"

    if st.session_state[feedback_submitted_key]:
        st.success(f"Thanks for your feedback ({st.session_state[feedback_submitted_key]})!")
        # Optionally, disable further feedback input here if desired
        # For example, by returning or disabling the widgets below.
        # For now, just showing the message.
        return

    cols = st.columns([1,1,5]) # Thumbs up, Thumbs down, Comment Area + Submit Button

    user_comment = cols[2].text_area("Optional comment:", key=comment_key, height=75,
                                     placeholder="Your thoughts on this item...",
                                     label_visibility="collapsed") # Hide label if caption is enough

    positive_pressed = cols[0].button("👍 Positive", key=f"positive_{item_type}_{item_id}_{key_suffix}")
    negative_pressed = cols[1].button("👎 Negative", key=f"negative_{item_type}_{item_id}_{key_suffix}")

    if positive_pressed:
        if orchestrator.store_user_feedback(item_id, item_type, "positive", user_comment, current_operation_mode, related_prompt_preview):
            st.session_state[feedback_submitted_key] = "Positive"
            st.rerun()
        else:
            st.error("Failed to store positive feedback.")

    if negative_pressed:
        if orchestrator.store_user_feedback(item_id, item_type, "negative", user_comment, current_operation_mode, related_prompt_preview):
            st.session_state[feedback_submitted_key] = "Negative"
            st.rerun()
        else:
            st.error("Failed to store negative feedback.")

def main():
   st.markdown("""<div style='text-align:center;background:linear-gradient(90deg,#FF6B6B,#4ECDC4,#45B7D1,#96CEB4);padding:20px;border-radius:10px;margin-bottom:20px'>
   <h1 style='color:white;text-shadow:2px 2px 4px rgba(0,0,0,0.5)'>TERMINUS AI NEXUS</h1>
   <p style='color:white;font-size:18px'>ULTIMATE LOCAL AI ECOSYSTEM | ADVANCED ORCHESTRATION</p></div>""",unsafe_allow_html=True)

   with st.sidebar:
       st.header("COMMAND CENTER")
       # Defined operation modes
       operation_modes = [
           "Multi-Agent Chat", "Knowledge Base Explorer", "Document Processing",
           "Web Intelligence", "Image Generation", "Video Processing",
           "Audio Processing", "Code Generation", "System Information",
           # "Data Analysis", "Creative Suite" # Keep commented if not fully implemented yet
       ]
       operation_mode = st.selectbox("Operation Mode", operation_modes, key="main_operation_mode")

       st.subheader("Agent Selection")
       # Ensure orchestrator and its agents list are loaded
       if hasattr(orchestrator, 'agents') and orchestrator.agents:
           agent_names = [a.name for a in orchestrator.agents if a.active] # Only show active agents
           default_selection = agent_names[:min(len(agent_names), 3)] # Default to first 3 active or fewer
           selected_agents_names = st.multiselect("Active Agents for Parallel Execution",
                                               options=agent_names,
                                               default=default_selection,
                                               key="sidebar_selected_agents")
       else:
           st.warning("Orchestrator agents not loaded. Agent selection unavailable.")
           selected_agents_names = [] # Fallback

       # Parameters (less critical if orchestrator not loaded, but good to have defaults)
       st.subheader("LLM Parameters (General)")
       temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, key="llm_temp") # Adjusted range and default
       # max_tokens is often model-specific, might be better handled by orchestrator or agent defaults
       # max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 128, key="llm_max_tokens")

   if operation_mode=="Multi-Agent Chat":
       st.subheader("💬 UNIVERSAL AI COMMAND (MULTI-AGENT CHAT)")

       # Initialize chat history in session state if it doesn't exist
       if "chat_messages" not in st.session_state:
           st.session_state.chat_messages = []

       # Display prior messages
       for message in st.session_state.chat_messages:
           with st.chat_message(message["role"]):
               st.markdown(message["content"])
               if message["role"] == "assistant" and message.get("is_plan_outcome", False):
                    # This is where feedback for the overall plan outcome is displayed
                    plan_feedback_item_id = message.get("feedback_item_id", str(uuid.uuid4())) # Use stored or generate
                    plan_feedback_item_type = message.get("feedback_item_type", "master_plan_outcome")
                    plan_related_prompt = message.get("related_user_prompt_for_feedback", "N/A")

                    display_feedback_widgets(
                        item_id=plan_feedback_item_id,
                        item_type=plan_feedback_item_type,
                        key_suffix=f"plan_outcome_hist_{plan_feedback_item_id}",
                        current_operation_mode=operation_mode,
                        related_prompt_preview=plan_related_prompt
                    )

       user_prompt = st.chat_input("Your command to the AI constellation...")
       use_master_planner = st.sidebar.checkbox("✨ Use MasterPlanner for complex requests", value=True, key="use_master_planner_toggle")

       if user_prompt:
           st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
           with st.chat_message("user"):
               st.markdown(user_prompt)

           with st.chat_message("assistant"):
               if use_master_planner:
                   with st.spinner("MasterPlanner is thinking and coordinating... This may take a while."):
                       # This call now returns the detailed step results AND the assistant summary
                       # The orchestrator.conversation_history is updated internally by execute_master_plan
                       plan_execution_step_results = asyncio.run(orchestrator.execute_master_plan(user_prompt))

                   # The assistant's summary is the last item in orchestrator's history
                   assistant_summary_turn = orchestrator.get_conversation_history_for_display()[-1] if orchestrator.get_conversation_history_for_display() else None

                   if assistant_summary_turn and assistant_summary_turn["role"] == "assistant":
                       assistant_response_content = assistant_summary_turn.get("content", "MasterPlanner finished.")
                       plan_log_kb_id = assistant_summary_turn.get("plan_log_kb_id") # Get the KB ID of the plan log

                       st.markdown(assistant_response_content) # Display summary in chat

                       # Store this summary in UI's session_state.chat_messages for redraws
                       # Also mark it as a plan outcome and store necessary IDs for feedback
                       st.session_state.chat_messages.append({
                           "role": "assistant",
                           "content": assistant_response_content,
                           "is_plan_outcome": True, # Mark this as a plan outcome message
                           "feedback_item_id": plan_log_kb_id if plan_log_kb_id else f"plan_{str(uuid.uuid4())}",
                           "feedback_item_type": "master_plan_log_outcome" if plan_log_kb_id else "master_plan_outcome_fallback",
                           "related_user_prompt_for_feedback": user_prompt # The prompt that initiated this plan
                       })

                       # Display detailed step results below the summary
                       if plan_execution_step_results:
                           st.markdown("--- \n**Master Plan Execution Details:**")
                           for i, step_result in enumerate(plan_execution_step_results):
                               status_icon = "✅" if step_result.get("status") == "success" else "⚠️" if step_result.get("status") == "info" else "❌"
                               exp_title = f"{status_icon} Step {step_result.get('step_id', i+1)}: {step_result.get('agent', 'N/A')}"
                               with st.expander(exp_title, expanded=(step_result.get("status") != "success")): # Expand errors by default
                                   st.markdown(f"**Status:** {step_result.get('status')}")
                                   st.markdown(f"**Response/Output:**")
                                   st.code(str(step_result.get('response', 'No textual response.'))) # Use st.code for better formatting of potentially long/structured responses

                                   # Display rich media from step_result if available
                                   if step_result.get("image_path"): st.image(step_result["image_path"])
                                   if step_result.get("frame_path"): st.image(step_result["frame_path"])
                                   if step_result.get("gif_path"): st.image(step_result["gif_path"])
                                   if step_result.get("speech_path"): st.audio(step_result["speech_path"])
                                   if step_result.get("modified_file"): st.info(f"File modified: {step_result.get('modified_file')}")

                                   with st.expander("Full JSON Details for this step", expanded=False):
                                       st.json(step_result)
                       else:
                           st.info("MasterPlanner did not return detailed step results or the plan was empty.")
                   else:
                       st.error("MasterPlanner finished, but could not retrieve a summary for display.")

               else: # Not using MasterPlanner, direct parallel execution
                   with st.spinner("Processing with selected agents..."):
                       current_context = {"current_mode": operation_mode, "user_prompt": user_prompt}
                       results = asyncio.run(orchestrator.parallel_execution(prompt=user_prompt, selected_agents_names=selected_agents_names, context=current_context))

                   # For non-planner mode, the "assistant" response is a collection of individual agent outputs
                   # We'll display them directly and add them to chat history individually for feedback.
                   if not results:
                       st.info("No results from agents.")
                       st.session_state.chat_messages.append({"role": "assistant", "content": "No results from agents."})
                   else:
                       combined_response_for_history = "Multiple agent responses:\n"
                       for result in results:
                           status_icon = "✅" if result.get("status") == "success" else "⚠️" if result.get("status") == "info" else "❌"
                           agent_name = result.get('agent', 'N/A')
                           response_text = str(result.get('response', 'No textual response.'))

                           # Display individual agent response
                           st.markdown(f"**{status_icon} {agent_name} ({result.get('model', 'N/A')}):**")
                           st.markdown(response_text)

                           # Add feedback for this individual agent response
                           response_item_id = f"agent_resp_{agent_name}_{str(uuid.uuid4())[:8]}"
                           display_feedback_widgets(
                               item_id=response_item_id,
                               item_type="agent_chat_response",
                               key_suffix=response_item_id, # Unique key for UI state
                               current_operation_mode=operation_mode,
                               related_prompt_preview=user_prompt[:200]
                           )
                           st.markdown("---")
                           combined_response_for_history += f"\n---\nAgent: {agent_name}\nStatus: {result.get('status')}\nResponse: {response_text[:200]}...\n---"

                       # Add a single summary of these responses to the main chat history for session state
                       # This is a simplified representation for the main history list.
                       st.session_state.chat_messages.append({"role": "assistant", "content": combined_response_for_history})
           st.rerun() # Rerun to update the chat display immediately


   elif operation_mode=="Document Processing":
       st.subheader("📄 UNIVERSAL DOCUMENT PROCESSOR")
       uploaded_files=st.file_uploader("Upload documents",accept_multiple_files=True,type=['pdf','docx','xlsx','txt','csv','json','html'], key="doc_proc_uploader")

       if uploaded_files:
           for file in uploaded_files:
               file_key = f"doc_{file.name}_{file.id if hasattr(file, 'id') else str(uuid.uuid4())[:8]}" # More unique key
               with st.expander(f"📄 {file.name}"):
                   try:
                       content=doc_processor.process_file(file) # Assumes process_file can handle UploadedFile
                       if isinstance(content, (dict, list)): # If JSON or CSV parsed into structure
                           st.json(content)
                           content_str_for_analysis = json.dumps(content)[:5000] # Analyze JSON string
                       else: # Text content
                           st.text_area("Content Preview", str(content)[:2000]+"..." if len(str(content))>2000 else str(content), height=200, key=f"preview_{file_key}")
                           content_str_for_analysis = str(content)[:5000]

                       if st.button(f"Analyze with AI & Store Excerpt in KB",key=f"analyze_{file_key}"):
                           with st.spinner("Storing document excerpt and analyzing..."):
                               doc_metadata = {
                                   "source": "document_upload", "filename": file.name,
                                   "filetype": file.type if file.type else Path(file.name).suffix,
                                   "processed_timestamp": datetime.datetime.now().isoformat()
                               }
                               excerpt_to_store = content_str_for_analysis[:2000] # Consistent excerpt length

                               async def store_and_publish_doc_excerpt_wrapper(): # Wrapper for async calls
                                   kb_store_result = await orchestrator.store_knowledge(content=excerpt_to_store, metadata=doc_metadata)
                                   if kb_store_result.get("status") == "success":
                                       st.success(f"Document excerpt stored in KB (ID: {kb_store_result.get('id')}).")
                                       await orchestrator.publish_message(
                                           message_type="kb.document_excerpt.added",
                                           source_agent_name="DocumentProcessorUI",
                                           payload={"kb_id": kb_store_result.get("id"), "filename": file.name}
                                       )
                                   else:
                                       st.error(f"Failed to store excerpt: {kb_store_result.get('message')}")

                                   # Proceed with AI analysis
                                   analysis_prompt = f"Analyze this document excerpt: {excerpt_to_store}"
                                   # Use selected_agents_names from sidebar
                                   analysis_results = await orchestrator.parallel_execution(analysis_prompt, selected_agents_names)
                                   for res in analysis_results:
                                       st.info(f"**{res.get('agent','N/A')}**: {str(res.get('response','N/A'))[:500]}...")

                               asyncio.run(store_and_publish_doc_excerpt_wrapper())
                   except Exception as e:
                       st.error(f"Error processing file {file.name}: {e}")


   elif operation_mode=="Web Intelligence":
       st.subheader("🌐 WEB INTELLIGENCE NEXUS")
       web_task = st.radio("Select Task:", ("Search Web & Analyze", "Scrape Single URL & Analyze"), key="web_intel_task_radio")

       if web_task == "Search Web & Analyze":
           search_query = st.text_input("Enter Search Query:", key="web_search_query")
           if st.button("Search & Analyze Results", key="web_search_button"):
               if search_query:
                   with st.spinner("Searching the web and analyzing results..."):
                       try:
                           search_results = web_intel.search_web(search_query)
                           st.success(f"Found {len(search_results)} results for '{search_query}'. Displaying top 3.")
                           st.json(search_results[:3]) # Show top 3 results

                           if search_results:
                               analysis_prompt_content = f"Analyze these web search results for query '{search_query}':\n"
                               for i, sr in enumerate(search_results[:3]): # Analyze top 3
                                   analysis_prompt_content += f"\nResult {i+1}:\nTitle: {sr.get('title')}\nURL: {sr.get('url')}\nSnippet: {sr.get('snippet')}\n---"

                               # Use selected_agents_names from sidebar
                               analysis_results = asyncio.run(orchestrator.parallel_execution(analysis_prompt_content, selected_agents_names))
                               st.subheader("AI Analysis of Search Results:")
                               for res in analysis_results:
                                   with st.expander(f"{res.get('agent','N/A')} Analysis"):
                                       st.write(str(res.get("response","N/A")))
                           else:
                               st.info("No search results to analyze.")
                       except Exception as e:
                           st.error(f"Error during web search & analysis: {e}")
               else:
                   st.warning("Please enter a search query.")

       elif web_task == "Scrape Single URL & Analyze":
           url_to_scrape = st.text_input("Enter URL to Scrape:", placeholder="https://example.com", key="web_scrape_url")
           if st.button("Scrape URL & Analyze Content", key="web_scrape_button"):
               if url_to_scrape:
                   with st.spinner(f"Scraping {url_to_scrape} and analyzing content..."):
                       # Orchestrator's execute_agent for WebCrawler handles scraping, summarization, and KB storage
                       # The prompt for WebCrawler agent is the URL itself.
                       webcrawler_agent = next((a for a in orchestrator.agents if a.name == "WebCrawler" and a.active), None)
                       if webcrawler_agent:
                           scrape_and_analyze_result = asyncio.run(orchestrator.execute_agent(webcrawler_agent, url_to_scrape))

                           if scrape_and_analyze_result.get("status") == "success":
                               st.success(f"Successfully processed URL: {url_to_scrape}")
                               st.markdown("**Summary/Response from WebCrawler:**")
                               st.markdown(scrape_and_analyze_result.get("response", "No summary available."))
                               if scrape_and_analyze_result.get("original_url"): # Should be same as url_to_scrape
                                   st.caption(f"Original URL: {scrape_and_analyze_result.get('original_url')}")
                               if "full_content_length" in scrape_and_analyze_result:
                                    st.caption(f"Full content length (approx.): {scrape_and_analyze_result.get('full_content_length')} chars")
                               # Info about KB storage is logged by orchestrator, UI can just confirm success.
                           else:
                               st.error(f"Failed to process URL: {scrape_and_analyze_result.get('response', 'Unknown error')}")
                       else:
                           st.error("WebCrawler agent is not available or active.")
               else:
                   st.warning("Please enter a URL to scrape.")


   elif operation_mode == "Image Generation":
       st.subheader("🎨 IMAGE GENERATION STUDIO")
       image_prompt = st.text_area("Enter your image description:", height=100, placeholder="E.g., 'A photorealistic cat astronaut on the moon'", key="img_gen_prompt")
       if st.button("Generate Image", type="primary", key="img_gen_button"):
           if image_prompt:
               with st.spinner("Generating image... This may take a while."):
                   imageforge_agent = next((a for a in orchestrator.agents if a.name == "ImageForge" and a.active), None)
                   if imageforge_agent:
                       result = asyncio.run(orchestrator.execute_agent(imageforge_agent, image_prompt))
                       if result.get("status") == "success" and result.get("image_path"):
                           st.image(result["image_path"], caption=f"Generated for: {image_prompt}")
                           st.success(f"Image saved to: {result['image_path']}")
                       else:
                           st.error(f"Image generation failed: {result.get('response', 'Unknown error')}")
                   else:
                       st.error("ImageForge agent not available/active.")
           else:
               st.warning("Please enter an image description.")

   elif operation_mode == "Video Processing": # Assuming methods exist in orchestrator
       st.subheader("🎞️ VIDEO PROCESSING UTILITIES")
       # ... (UI for video tasks: get_video_metadata, extract_video_frame, convert_video_to_gif) ...
       # This section would be similar to the one in the user's provided script, calling orchestrator methods.
       # For brevity, detailed UI for this is omitted here but would follow the pattern.
       st.info("Video processing UI placeholder. Refer to the full script for detailed implementation.")


   elif operation_mode == "Audio Processing": # Assuming methods exist in orchestrator
       st.subheader("🎤 AUDIO PROCESSING SUITE")
       # ... (UI for audio tasks: get_audio_info, convert_audio_format, text_to_speech) ...
       st.info("Audio processing UI placeholder.")

   elif operation_mode == "Code Generation": # Assuming methods exist in orchestrator
       st.subheader("💻 PROJECT SCAFFOLDING & CODE GENERATION")
       # ... (UI for scaffolding, AI code modification, explanation, module generation) ...
       st.info("Code generation UI placeholder.")

   elif operation_mode == "System Information":
       st.subheader("📊 SYSTEM INFORMATION DASHBOARD")
       sys_admin_agent = next((a for a in orchestrator.agents if a.name == "SystemAdmin" and a.active), None)
       if not sys_admin_agent:
           st.error("SystemAdmin agent not available or active.")
           return

       sys_info_tasks = {
           "OS Details": "os_info", "CPU Details": "cpu_info",
           "Disk Space": "disk_space", "Memory Usage": "memory_usage",
           "Network Config": "network_config"
       }
       cols = st.columns(len(sys_info_tasks))
       for i, (label, cmd_key) in enumerate(sys_info_tasks.items()):
           if cols[i].button(label, key=f"sysinfo_{cmd_key}_btn"):
               with st.spinner(f"Fetching {label}..."):
                   # Using execute_agent with a prompt that maps to get_system_info's keys
                   # The prompt for SystemAdmin now needs to be specific to trigger the right get_system_info key
                   # E.g., "get disk space", "get os info"
                   descriptive_prompt = f"get {cmd_key.replace('_', ' ')}" # Construct a descriptive prompt
                   result = asyncio.run(orchestrator.execute_agent(sys_admin_agent, descriptive_prompt))
                   with st.expander(f"{label} Output", expanded=True):
                       if result.get("status") == "success":
                           if isinstance(result.get("data"), dict): st.json(result.get("data"))
                           else: st.text(str(result.get("data","N/A")))
                       else: st.error(result.get("message", "Failed to fetch info."))

       st.markdown("---")
       top_n = st.number_input("Number of processes for 'Top Processes':", 1, 50, 10, key="sysinfo_top_n_input")
       if st.button("Get Top Processes", key="sysinfo_top_proc_btn"):
           with st.spinner(f"Fetching top {top_n} processes..."):
               result = asyncio.run(orchestrator.execute_agent(sys_admin_agent, f"top {top_n} processes"))
               with st.expander(f"Top {top_n} Processes Output", expanded=True):
                   if result.get("status") == "success": st.text(str(result.get("data","N/A")))
                   else: st.error(result.get("message", "Failed to fetch processes."))

       st.markdown("---")
       st.subheader("Feedback Analysis")
       if st.button("📊 Generate & Store Feedback Analysis Report", key="gen_feedback_report_button_ui"):
            with st.spinner("Generating feedback analysis report..."):
                report_result = asyncio.run(orchestrator.generate_and_store_feedback_report())
                if report_result.get("status") == "success":
                    st.success(report_result.get("message", "Report generated and stored!"))
                    if report_result.get("kb_id"): st.caption(f"Stored in KB with ID: {report_result.get('kb_id')}")
                else:
                    st.error(report_result.get("message", "Failed to generate/store report."))


   elif operation_mode == "Knowledge Base Explorer":
       st.subheader("🧠 KNOWLEDGE BASE EXPLORER")
       query_text = st.text_input("Search query for Knowledge Base:", key="kb_explorer_query")

       col1_kb, col2_kb, col3_kb = st.columns([2,1,1])
       n_results_kb = col1_kb.number_input("Num results:", 1, 50, 5, key="kb_explorer_n_results")
       filter_key_kb = col2_kb.text_input("Filter key (optional):", key="kb_explorer_fkey")
       filter_value_kb = col3_kb.text_input("Filter value (optional):", key="kb_explorer_fval")

       if st.button("Search Knowledge Base", key="kb_explorer_search_btn"):
           if query_text:
               filter_dict_kb = {filter_key_kb.strip(): filter_value_kb.strip()} if filter_key_kb and filter_value_kb else None
               with st.spinner("Searching KB..."):
                   kb_resp = asyncio.run(orchestrator.retrieve_knowledge(query_text, n_results_kb, filter_dict_kb))
               if kb_resp.get("status") == "success":
                   results_kb = kb_resp.get("results", [])
                   st.info(f"Found {len(results_kb)} results.")
                   for i, item_kb in enumerate(results_kb):
                       item_id_for_feedback = item_kb.get('id', f"kb_item_{i}_{str(uuid.uuid4())[:4]}")
                       with st.expander(f"Result {i+1}: ID `{item_id_for_feedback}` (Distance: {item_kb.get('distance', -1):.4f})"):
                           st.text_area("Content:", str(item_kb.get('document','N/A')), height=150, disabled=True, key=f"kb_item_doc_{item_id_for_feedback}")
                           st.json(item_kb.get('metadata', {}))

                           display_feedback_widgets(
                               item_id=item_id_for_feedback,
                               item_type="kb_search_result_item",
                               key_suffix=f"kb_item_fb_{item_id_for_feedback}",
                               current_operation_mode=operation_mode,
                               related_prompt_preview=query_text[:200]
                           )
               else:
                   st.error(f"KB Search Failed: {kb_resp.get('message', 'Unknown error')}")
           else:
               st.warning("Please enter a search query for the KB.")


if __name__=="__main__":
   # Ensure an event loop exists for Streamlit context if not already running
   try:
       loop = asyncio.get_running_loop()
   except RuntimeError:  # 'RuntimeError: There is no current event loop...'
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)

   # Check if orchestrator was loaded successfully before running main
   if 'orchestrator' in globals() and orchestrator is not None:
       main()
   else:
       # This message will be displayed if the import of orchestrator failed at the top.
       # The st.error and st.stop() at the import location should handle this.
       print("UI cannot start because core orchestrator components failed to load.")
