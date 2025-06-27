import streamlit as st,asyncio,json,os,subprocess,time,datetime,uuid # Ensure time, datetime, and uuid are imported
from pathlib import Path # Path is used for temp_video_path
import pandas as pd,plotly.express as px,plotly.graph_objects as go # Keep Plotly for potential future Data Analysis UI
from typing import Optional # For type hinting in display_feedback_widgets

# Assuming master_orchestrator.py (and thus its instances) are in $INSTALL_DIR/agents/
# and terminus_ui.py is in $INSTALL_DIR/
# The launch_terminus.py script should handle adding $INSTALL_DIR to sys.path or PYTHONPATH
# so that `from agents.master_orchestrator import ...` works.
try:
    # doc_processor has been removed. web_intel will be handled in a future step.
    from agents.master_orchestrator import orchestrator, web_intel
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
                       # Call the new orchestrator method for content extraction
                       extraction_result = asyncio.run(orchestrator.extract_document_content(file, file.name))
                       content = extraction_result.get("content")
                       status = extraction_result.get("status")
                       message = extraction_result.get("message")
                       file_type_processed = extraction_result.get("file_type_processed", "unknown")

                       content_str_for_analysis = "" # Initialize

                       if status == "success":
                           st.info(f"Successfully processed '{file.name}' ({file_type_processed}). {message if message else ''}")
                           if content:
                               if file_type_processed == 'json':
                                   try:
                                       st.json(json.loads(content)) # Parse stringified JSON for st.json
                                   except json.JSONDecodeError:
                                       st.text_area("Content Preview (JSON parse error, showing raw):", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}")
                               else:
                                   st.text_area("Content Preview", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}")
                               content_str_for_analysis = content[:5000] # Use up to 5000 chars for analysis excerpt
                           else: # Success but no content (e.g. image-only PDF)
                               st.info(f"Successfully processed '{file.name}', but no text content was extracted. {message if message else ''}")

                       elif status == "partial_success":
                           st.warning(f"Partially processed '{file.name}' ({file_type_processed}). {message if message else ''}")
                           if content:
                               st.text_area("Content Preview (Partial):", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}")
                               content_str_for_analysis = content[:5000]

                       else: # status == "error" or unknown
                           st.error(f"Error processing file {file.name}: {message}")
                           # content_str_for_analysis remains empty

                       # "Analyze with AI & Store Excerpt in KB" button logic
                       # This button should only be active if content_str_for_analysis has something.
                       if content_str_for_analysis:
                           if st.button(f"Analyze with AI & Store Excerpt in KB",key=f"analyze_{file_key}"):
                               with st.spinner("Storing document excerpt and analyzing..."):
                                   doc_metadata = {
                                       "source": "document_upload", "filename": file.name,
                                       "filetype": file_type_processed, # Use detected file type
                                       "processed_timestamp": datetime.datetime.now().isoformat(),
                                       "extraction_status": status,
                                       "extraction_message": message
                                   }
                                   # Ensure excerpt_to_store is not empty if button is active
                                   excerpt_to_store = content_str_for_analysis[:2000] if content_str_for_analysis else "No content extracted for analysis."


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

   elif operation_mode == "Video Processing":
       st.subheader("🎞️ VIDEO PROCESSING UTILITIES")

       uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")

       if uploaded_video is not None:
           # Save uploaded video to a temporary path as orchestrator methods expect paths
           temp_dir = Path(orchestrator.video_processing_dir) / "temp_uploads"
           temp_dir.mkdir(parents=True, exist_ok=True)
           temp_video_path = temp_dir / uploaded_video.name

           with open(temp_video_path, "wb") as f:
               f.write(uploaded_video.getbuffer())

           st.video(str(temp_video_path))

           st.markdown("---")
           st.subheader("Get Video Metadata")
           if st.button("Extract Metadata", key="video_meta_button"):
               with st.spinner("Extracting video metadata..."):
                   result = asyncio.run(orchestrator.get_video_metadata(str(temp_video_path)))
                   if result.get("status") == "success":
                       st.json(result.get("metadata"))
                       display_feedback_widgets(f"meta_{uploaded_video.name}", "video_metadata_result", f"meta_{uploaded_video.name}_fb", operation_mode)
                   else:
                       st.error(f"Failed to get metadata: {result.get('message')}")

           st.markdown("---")
           st.subheader("Extract Frame")
           extract_ts = st.text_input("Timestamp for frame (e.g., 00:00:05 or 5.0)", key="video_extract_ts")
           if st.button("Extract Frame", key="video_extract_frame_button"):
               if extract_ts:
                   with st.spinner("Extracting frame..."):
                       result = asyncio.run(orchestrator.extract_video_frame(str(temp_video_path), extract_ts))
                       if result.get("status") == "success" and result.get("frame_path"):
                           st.image(result.get("frame_path"), caption=f"Frame at {extract_ts}")
                           st.success(f"Frame saved to: {result.get('frame_path')}")
                           display_feedback_widgets(f"frame_{uploaded_video.name}_{extract_ts}", "video_frame_extraction_result", f"frame_{uploaded_video.name}_{extract_ts}_fb", operation_mode)
                       else:
                           st.error(f"Failed to extract frame: {result.get('message')}")
               else:
                   st.warning("Please enter a timestamp.")

           st.markdown("---")
           st.subheader("Convert to GIF")
           gif_start_ts = st.text_input("GIF Start Timestamp (e.g., 00:00:03 or 3.0)", key="video_gif_start_ts")
           gif_end_ts = st.text_input("GIF End Timestamp (e.g., 00:00:08 or 8.0)", key="video_gif_end_ts")
           # gif_scale = st.slider("Resolution Scale", 0.1, 1.0, 0.5, 0.05, key="video_gif_scale") # Optional
           # gif_fps = st.number_input("GIF FPS", 1, 30, 10, key="video_gif_fps") # Optional

           if st.button("Convert to GIF", key="video_convert_gif_button"):
               if gif_start_ts and gif_end_ts:
                   with st.spinner("Converting to GIF..."):
                       # Using default scale and fps for simplicity in this step
                       result = asyncio.run(orchestrator.convert_video_to_gif(str(temp_video_path), gif_start_ts, gif_end_ts))
                       if result.get("status") == "success" and result.get("gif_path"):
                           st.image(result.get("gif_path"), caption=f"GIF from {gif_start_ts} to {gif_end_ts}")
                           st.success(f"GIF saved to: {result.get('gif_path')}")
                           display_feedback_widgets(f"gif_{uploaded_video.name}_{gif_start_ts}_{gif_end_ts}", "video_gif_conversion_result", f"gif_{uploaded_video.name}_{gif_start_ts}_{gif_end_ts}_fb", operation_mode)
                       else:
                           st.error(f"Failed to convert to GIF: {result.get('message')}")
               else:
                   st.warning("Please enter both start and end timestamps for GIF conversion.")

           # Clean up temp file after processing for this session (or manage more robustly if needed)
           # For now, simple removal. This means re-upload is needed for new operations on same video.
           # Consider leaving it and having a clear button or session-based cleanup.
           # For simplicity, let's assume it's fine for now.
           # if temp_video_path.exists():
           #     temp_video_path.unlink()

   elif operation_mode == "Audio Processing":
       st.subheader("🎤 AUDIO PROCESSING SUITE")

       audio_task = st.radio("Select Audio Task:",
                             ("Get Audio Info", "Convert Audio Format", "Text-to-Speech"),
                             key="audio_task_radio")

       if audio_task == "Text-to-Speech":
           st.markdown("---")
           tts_text = st.text_area("Text to convert to speech:", key="tts_text_input", height=150)
           tts_filename_stem = st.text_input("Output filename stem (optional):", value="speech_output", key="tts_filename_stem")
           if st.button("Generate Speech", key="tts_generate_button"):
               if tts_text.strip():
                   with st.spinner("Generating speech..."):
                       result = asyncio.run(orchestrator.text_to_speech(tts_text, tts_filename_stem))
                       if result.get("status") == "success" and result.get("speech_path"):
                           st.audio(result.get("speech_path"), format='audio/mp3')
                           st.success(f"Speech saved to: {result.get('speech_path')}")
                           display_feedback_widgets(f"tts_{tts_filename_stem}", "tts_result", f"tts_{tts_filename_stem}_fb", operation_mode, tts_text[:100])
                       else:
                           st.error(f"Failed to generate speech: {result.get('message')}")
               else:
                   st.warning("Please enter text for speech conversion.")
       else: # Get Audio Info or Convert Audio Format
           st.markdown("---")
           uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a"], key="audio_uploader")
           if uploaded_audio is not None:
               # Save uploaded audio to a temporary path
               temp_audio_dir = Path(orchestrator.audio_processing_dir) / "temp_uploads"
               temp_audio_dir.mkdir(parents=True, exist_ok=True)
               temp_audio_path = temp_audio_dir / uploaded_audio.name
               with open(temp_audio_path, "wb") as f:
                   f.write(uploaded_audio.getbuffer())

               st.audio(str(temp_audio_path))

               if audio_task == "Get Audio Info":
                   st.markdown("---")
                   if st.button("Get Audio Info", key="audio_info_button"):
                       with st.spinner("Extracting audio information..."):
                           result = asyncio.run(orchestrator.get_audio_info(str(temp_audio_path)))
                           if result.get("status") == "success":
                               st.json(result.get("info"))
                               display_feedback_widgets(f"info_{uploaded_audio.name}", "audio_info_result", f"info_{uploaded_audio.name}_fb", operation_mode)
                           else:
                               st.error(f"Failed to get audio info: {result.get('message')}")

               elif audio_task == "Convert Audio Format":
                   st.markdown("---")
                   target_format = st.selectbox("Target Format:", ["mp3", "wav", "ogg", "flac"], key="audio_convert_format_select")
                   if st.button("Convert Format", key="audio_convert_button"):
                       with st.spinner(f"Converting to {target_format}..."):
                           result = asyncio.run(orchestrator.convert_audio_format(str(temp_audio_path), target_format))
                           if result.get("status") == "success" and result.get("output_path"):
                               st.success(f"Converted audio saved to: {result.get('output_path')}")
                               try: # Attempt to display the converted audio
                                   st.audio(result.get("output_path"))
                               except Exception as e_audio_display:
                                   st.warning(f"Could not display converted audio directly: {e_audio_display}. Please check the file at the path provided.")
                               display_feedback_widgets(f"convert_{uploaded_audio.name}_{target_format}", "audio_conversion_result", f"convert_{uploaded_audio.name}_{target_format}_fb", operation_mode)
                           else:
                               st.error(f"Failed to convert audio: {result.get('message')}")

               # Consider temp file cleanup logic here or at session end
               # if temp_audio_path.exists():
               #     temp_audio_path.unlink()
   elif operation_mode == "Code Generation":
       st.subheader("💻 PROJECT SCAFFOLDING & CODE GENERATION")

       code_task = st.selectbox("Select Code Task:",
                                ["Scaffold New Project", "Explain Code Snippet", "Generate Code Module/Class"],
                                key="code_task_select")

       if code_task == "Scaffold New Project":
           st.markdown("---")
           st.write("Scaffold a new project structure using AutoDev.")
           project_name = st.text_input("Project Name:", key="scaffold_project_name")
           project_type_options = ["python_cli", "streamlit_dashboard", "nodejs_api"] # Add more as AutoDev supports them
           project_type = st.selectbox("Project Type:", project_type_options, key="scaffold_project_type")

           if st.button("Scaffold Project", key="scaffold_project_button"):
               if project_name and project_type:
                   with st.spinner(f"Scaffolding '{project_name}' ({project_type})..."):
                       # This call is synchronous in the current auto_dev.py, but orchestrator method is async
                       result = asyncio.run(orchestrator.scaffold_new_project(project_name, project_type))
                       if result.get("status") == "success":
                           st.success(result.get("message"))
                           display_feedback_widgets(f"scaffold_{project_name}", "project_scaffolding_result", f"scaffold_{project_name}_fb", operation_mode, f"Scaffold: {project_name} ({project_type})")
                       else:
                           st.error(f"Failed to scaffold project: {result.get('message')}")
               else:
                   st.warning("Please provide both project name and type.")

       elif code_task == "Explain Code Snippet":
           st.markdown("---")
           st.write("Get an AI-powered explanation for a code snippet.")
           code_snippet_explain = st.text_area("Code Snippet to Explain:", height=200, key="explain_code_snippet_input")
           # language_explain = st.text_input("Language (e.g., python, javascript):", value="python", key="explain_code_language") # Optional: make it a selectbox

           if st.button("Explain Snippet", key="explain_snippet_button"):
               if code_snippet_explain.strip():
                   with st.spinner("Generating explanation..."):
                       # Assuming orchestrator.explain_code_snippet exists and handles language if needed
                       result = asyncio.run(orchestrator.explain_code_snippet(code_snippet_explain)) # Add language if method supports
                       if result.get("status") == "success":
                           st.markdown("**Explanation:**")
                           st.markdown(result.get("explanation"))
                           display_feedback_widgets(f"explain_{hash(code_snippet_explain)}", "code_explanation_result", f"explain_{hash(code_snippet_explain)}_fb", operation_mode, code_snippet_explain[:100])
                       else:
                           st.error(f"Failed to get explanation: {result.get('message')}")
               else:
                   st.warning("Please enter a code snippet to explain.")

       elif code_task == "Generate Code Module/Class":
           st.markdown("---")
           st.write("Generate a new code module or class based on requirements.")
           requirements_generate = st.text_area("Requirements/Description for the module/class:", height=150, key="generate_module_requirements")
           # language_generate = st.text_input("Language (e.g., python):", value="python", key="generate_module_language") # Optional

           if st.button("Generate Module", key="generate_module_button"):
               if requirements_generate.strip():
                   with st.spinner("Generating code module..."):
                       # Assuming orchestrator.generate_code_module exists
                       result = asyncio.run(orchestrator.generate_code_module(requirements_generate)) # Add language if method supports
                       if result.get("status") == "success":
                           st.markdown("**Generated Code:**")
                           st.code(result.get("generated_code"), language="python") # Assume python for now, or use language_generate
                           display_feedback_widgets(f"genmodule_{hash(requirements_generate)}", "code_module_generation_result", f"genmodule_{hash(requirements_generate)}_fb", operation_mode, requirements_generate[:100])
                       else:
                           st.error(f"Failed to generate module: {result.get('message')}")
               else:
                   st.warning("Please provide requirements for the module.")

       # Placeholder for "AI-Assisted Code Modification" - to be implemented if desired
       # st.markdown("---")
       # st.subheader("AI-Assisted Code Modification (Experimental)")
       # st.info("This feature is experimental. Always review changes carefully.")
       # modify_project_name = st.text_input("Project Name (must exist in AutoDev projects folder):", key="modify_project_name")
       # modify_file_path = st.text_input("Relative Path to File (e.g., src/main.py):", key="modify_file_path")
       # modify_instruction = st.text_area("Modification Instruction:", height=100, key="modify_instruction")
       # if st.button("Attempt Code Modification", key="modify_code_button"):
       #     # ... call orchestrator.modify_code_in_project ...
       #     pass
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
