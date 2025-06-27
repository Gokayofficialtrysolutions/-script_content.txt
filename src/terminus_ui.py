import streamlit as st,asyncio,json,os,subprocess,time,datetime,uuid # Ensure time, datetime, and uuid are imported
from pathlib import Path # Path is used for temp_video_path
import pandas as pd,plotly.express as px,plotly.graph_objects as go # Keep Plotly for potential future Data Analysis UI
from typing import Optional, List, Dict # Added List, Dict for type hinting

# Assuming master_orchestrator.py (and thus its instances) are in $INSTALL_DIR/agents/
# and terminus_ui.py is in $INSTALL_DIR/
# The launch_terminus.py script should handle adding $INSTALL_DIR to sys.path or PYTHONPATH
# so that `from agents.master_orchestrator import ...` works.
try:
    # doc_processor and web_intel have been removed.
    from agents.master_orchestrator import orchestrator
except ImportError as e:
    st.error(f"Critical Error: Could not import orchestrator components: {e}. UI cannot function.")
    st.stop() # Halt execution of the UI if core components can't be imported


st.set_page_config(page_title="TERMINUS AI",layout="wide",initial_sidebar_state="expanded")

# Helper function for feedback widgets (remains global)
def display_feedback_widgets(item_id: str, item_type: str,
                             key_suffix: str, # To ensure unique widget keys
                             current_operation_mode: str, # Renamed from operation_mode to avoid conflict
                             related_prompt_preview: Optional[str] = None):
    if 'orchestrator' not in globals() or orchestrator is None:
        st.warning("Feedback system unavailable: Orchestrator not loaded.")
        return
    st.write("---"); st.caption("Rate this response/item:")
    feedback_submitted_key = f"feedback_submitted_{item_type}_{item_id}_{key_suffix}"
    comment_key = f"comment_{item_type}_{item_id}_{key_suffix}"
    if feedback_submitted_key not in st.session_state: st.session_state[feedback_submitted_key] = None
    if st.session_state[feedback_submitted_key]: st.success(f"Thanks for your feedback ({st.session_state[feedback_submitted_key]})!"); return
    cols = st.columns([1,1,5])
    user_comment = cols[2].text_area("Optional comment:", key=comment_key, height=75, placeholder="Your thoughts on this item...", label_visibility="collapsed")
    if cols[0].button("👍 Positive", key=f"positive_{item_type}_{item_id}_{key_suffix}"):
        if orchestrator.store_user_feedback(item_id, item_type, "positive", user_comment, current_operation_mode, related_prompt_preview):
            st.session_state[feedback_submitted_key] = "Positive"; st.rerun()
        else: st.error("Failed to store positive feedback.")
    if cols[1].button("👎 Negative", key=f"negative_{item_type}_{item_id}_{key_suffix}"):
        if orchestrator.store_user_feedback(item_id, item_type, "negative", user_comment, current_operation_mode, related_prompt_preview):
            st.session_state[feedback_submitted_key] = "Negative"; st.rerun()
        else: st.error("Failed to store negative feedback.")

def cleanup_temp_uploads(base_path: Path, temp_folder_name: str = "temp_uploads"):
    temp_dir = base_path / temp_folder_name
    if temp_dir.exists() and temp_dir.is_dir():
        for item in temp_dir.iterdir():
            try:
                if item.is_file(): item.unlink()
            except Exception as e: print(f"Warning: Could not delete temp item {item}: {e}")

# --- UI Rendering Functions for Each Operation Mode ---

def render_multi_agent_chat_ui(op_mode: str, sel_agents: List[str], temp: float): # Added temp for completeness, though not directly used here yet
    st.subheader("💬 UNIVERSAL AI COMMAND (MULTI-AGENT CHAT)")
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("is_plan_outcome", False):
                display_feedback_widgets(
                    item_id=message.get("feedback_item_id", str(uuid.uuid4())),
                    item_type=message.get("feedback_item_type", "master_plan_outcome"),
                    key_suffix=f"plan_outcome_hist_{message.get('feedback_item_id', str(uuid.uuid4()))}",
                    current_operation_mode=op_mode,
                    related_prompt_preview=message.get("related_user_prompt_for_feedback", "N/A")
                )
    user_prompt = st.chat_input("Your command to the AI constellation...")
    use_master_planner = st.sidebar.checkbox("✨ Use MasterPlanner for complex requests", value=True, key="use_master_planner_toggle_chat") # Unique key

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.markdown(user_prompt)
        with st.chat_message("assistant"):
            if use_master_planner:
                with st.spinner("MasterPlanner is thinking..."):
                    plan_execution_step_results = asyncio.run(orchestrator.execute_master_plan(user_prompt)) # temp is not directly passed here, MasterPlanner uses its own defaults or gets from agent config
                assistant_summary_turn = orchestrator.get_conversation_history_for_display()[-1] if orchestrator.get_conversation_history_for_display() else None
                if assistant_summary_turn and assistant_summary_turn["role"] == "assistant":
                    # ... (rest of MasterPlanner result display logic - unchanged)
                    assistant_response_content = assistant_summary_turn.get("content", "MasterPlanner finished.")
                    plan_log_kb_id = assistant_summary_turn.get("plan_log_kb_id")
                    st.markdown(assistant_response_content)
                    st.session_state.chat_messages.append({
                       "role": "assistant", "content": assistant_response_content, "is_plan_outcome": True,
                       "feedback_item_id": plan_log_kb_id if plan_log_kb_id else f"plan_{str(uuid.uuid4())}",
                       "feedback_item_type": "master_plan_log_outcome" if plan_log_kb_id else "master_plan_outcome_fallback",
                       "related_user_prompt_for_feedback": user_prompt
                    })
                    if plan_execution_step_results:
                       st.markdown("--- \n**Master Plan Execution Details:**")
                       for i, step_result in enumerate(plan_execution_step_results):
                           status_icon = "✅" if step_result.get("status") == "success" else "⚠️" if step_result.get("status") == "info" else "❌"
                           exp_title = f"{status_icon} Step {step_result.get('step_id', i+1)}: {step_result.get('agent', 'N/A')}"
                           with st.expander(exp_title, expanded=(step_result.get("status") != "success")):
                               st.markdown(f"**Status:** {step_result.get('status')}")
                               st.markdown(f"**Response/Output:**"); st.code(str(step_result.get('response', 'No textual response.')))
                               if step_result.get("image_path"): st.image(step_result["image_path"]) # ... and other rich media
                               with st.expander("Full JSON Details for this step", expanded=False): st.json(step_result)
                    else: st.info("MasterPlanner did not return detailed step results or the plan was empty.")
                else: st.error("MasterPlanner finished, but could not retrieve a summary for display.")
            else: # Direct parallel execution
                with st.spinner("Processing..."):
                    # Note: orchestrator.parallel_execution itself would need to be aware of global `temperature` or take it as param if needed by _ollama_generate
                    # For now, assuming _ollama_generate within parallel_execution uses its own defaults or agent-specific settings.
                    results = asyncio.run(orchestrator.parallel_execution(prompt=user_prompt, selected_agents_names=sel_agents, context={"current_mode": op_mode, "user_prompt": user_prompt}))
                if not results: # ... (rest of parallel execution display logic - unchanged)
                    st.info("No results from agents."); st.session_state.chat_messages.append({"role": "assistant", "content": "No results from agents."})
                else:
                    combined_response_for_history = "Multiple agent responses:\n"
                    for result in results:
                        status_icon = "✅" if result.get("status") == "success" else "⚠️" if result.get("status") == "info" else "❌"
                        agent_name = result.get('agent', 'N/A'); response_text = str(result.get('response', 'No textual response.'))
                        st.markdown(f"**{status_icon} {agent_name} ({result.get('model', 'N/A')}):**"); st.markdown(response_text)
                        response_item_id = f"agent_resp_{agent_name}_{str(uuid.uuid4())[:8]}"
                        display_feedback_widgets(response_item_id, "agent_chat_response", response_item_id, op_mode, user_prompt[:200])
                        st.markdown("---"); combined_response_for_history += f"\n---\nAgent: {agent_name}\nStatus: {result.get('status')}\nResponse: {response_text[:200]}...\n---"
                    st.session_state.chat_messages.append({"role": "assistant", "content": combined_response_for_history})
        st.rerun()

def render_knowledge_base_explorer_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("🧠 KNOWLEDGE BASE EXPLORER")
    # ... (all logic from the original "Knowledge Base Explorer" elif block) ...
    query_text = st.text_input("Search query for Knowledge Base:", key="kb_explorer_query")
    col1_kb, col2_kb, col3_kb = st.columns([2,1,1])
    n_results_kb = col1_kb.number_input("Num results:", 1, 50, 5, key="kb_explorer_n_results")
    filter_key_kb = col2_kb.text_input("Filter key (optional):", key="kb_explorer_fkey")
    filter_value_kb = col3_kb.text_input("Filter value (optional):", key="kb_explorer_fval")
    if st.button("Search Knowledge Base", key="kb_explorer_search_btn"):
        if query_text:
            filter_dict_kb = {filter_key_kb.strip(): filter_value_kb.strip()} if filter_key_kb and filter_value_kb else None
            with st.spinner("Searching KB..."): kb_resp = asyncio.run(orchestrator.retrieve_knowledge(query_text, n_results_kb, filter_dict_kb))
            if kb_resp.get("status") == "success":
                results_kb = kb_resp.get("results", [])
                st.info(f"Found {len(results_kb)} results.")
                for i, item_kb in enumerate(results_kb):
                    item_id_for_feedback = item_kb.get('id', f"kb_item_{i}_{str(uuid.uuid4())[:4]}")
                    with st.expander(f"Result {i+1}: ID `{item_id_for_feedback}` (Distance: {item_kb.get('distance', -1):.4f})"):
                        st.text_area("Content:", str(item_kb.get('document','N/A')), height=150, disabled=True, key=f"kb_item_doc_{item_id_for_feedback}")
                        st.json(item_kb.get('metadata', {}))
                        display_feedback_widgets(item_id_for_feedback, "kb_search_result_item", f"kb_item_fb_{item_id_for_feedback}", op_mode, query_text[:200])
            else: st.error(f"KB Search Failed: {kb_resp.get('message', 'Unknown error')}")
        else: st.warning("Please enter a search query for the KB.")

def render_document_processing_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("📄 UNIVERSAL DOCUMENT PROCESSOR")
    # ... (all logic from the original "Document Processing" elif block, using op_mode, sel_agents as needed) ...
    uploaded_files=st.file_uploader("Upload documents",accept_multiple_files=True,type=['pdf','docx','xlsx','txt','csv','json','html'], key="doc_proc_uploader_rp") # Unique key
    if uploaded_files:
        for file in uploaded_files:
            file_key = f"doc_rp_{file.name}_{file.id if hasattr(file, 'id') else str(uuid.uuid4())[:8]}"
            with st.expander(f"📄 {file.name}"):
                try:
                    extraction_result = asyncio.run(orchestrator.extract_document_content(file, file.name))
                    content = extraction_result.get("content"); status = extraction_result.get("status"); message = extraction_result.get("message"); file_type_processed = extraction_result.get("file_type_processed", "unknown")
                    content_str_for_analysis = ""
                    if status == "success":
                        st.info(f"Successfully processed '{file.name}' ({file_type_processed}). {message if message else ''}")
                        if content:
                            if file_type_processed == 'json':
                                try: st.json(json.loads(content))
                                except json.JSONDecodeError: st.text_area("Content Preview (JSON parse error, showing raw):", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}")
                            else: st.text_area("Content Preview", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}")
                            content_str_for_analysis = content[:5000]
                        else: st.info(f"Successfully processed '{file.name}', but no text content was extracted. {message if message else ''}")
                    elif status == "partial_success":
                        st.warning(f"Partially processed '{file.name}' ({file_type_processed}). {message if message else ''}")
                        if content: st.text_area("Content Preview (Partial):", content[:5000]+"..." if len(content)>5000 else content, height=200, key=f"preview_{file_key}"); content_str_for_analysis = content[:5000]
                    else: st.error(f"Error processing file {file.name}: {message}")
                    if content_str_for_analysis:
                        if st.button(f"Analyze with AI & Store Excerpt in KB",key=f"analyze_{file_key}"):
                            with st.spinner("Storing and analyzing..."):
                                doc_metadata = {"source": "document_upload", "filename": file.name, "filetype": file_type_processed, "processed_timestamp": datetime.datetime.now().isoformat(), "extraction_status": status, "extraction_message": message}
                                excerpt_to_store = content_str_for_analysis[:2000] if content_str_for_analysis else "No content extracted."
                                async def store_and_publish_doc_excerpt_wrapper():
                                    kb_store_result = await orchestrator.store_knowledge(content=excerpt_to_store, metadata=doc_metadata)
                                    if kb_store_result.get("status") == "success": st.success(f"Excerpt stored in KB (ID: {kb_store_result.get('id')})."); await orchestrator.publish_message("kb.document_excerpt.added", "DocumentProcessorUI", {"kb_id": kb_store_result.get("id"), "filename": file.name})
                                    else: st.error(f"Failed to store excerpt: {kb_store_result.get('message')}")
                                    analysis_prompt = f"Analyze this document excerpt: {excerpt_to_store}"
                                    analysis_results = await orchestrator.parallel_execution(analysis_prompt, sel_agents) # Use sel_agents
                                    for res_a in analysis_results: st.info(f"**{res_a.get('agent','N/A')}**: {str(res_a.get('response','N/A'))[:500]}...")
                                asyncio.run(store_and_publish_doc_excerpt_wrapper())
                except Exception as e: st.error(f"Error processing file {file.name}: {e}")

def render_web_intelligence_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("🌐 WEB INTELLIGENCE NEXUS")
    # ... (all logic from the original "Web Intelligence" elif block, using op_mode, sel_agents as needed) ...
    web_task = st.radio("Select Task:", ("Search Web & Analyze", "Scrape Single URL & Analyze"), key="web_intel_task_radio_rp") # Unique key
    if web_task == "Search Web & Analyze":
        search_query = st.text_input("Enter Search Query:", key="web_search_query_rp")
        num_results_display = st.number_input("Number of search results to display/analyze:", 1, 10, 3, key="web_search_num_display_rp")
        if st.button("Search & Analyze Results", key="web_search_button_rp"):
            if search_query:
                with st.spinner("Searching the web..."):
                    webcrawler_agent = next((a for a in orchestrator.agents if a.name == "WebCrawler" and a.active), None)
                    search_results = []
                    if webcrawler_agent:
                        search_op_result = asyncio.run(orchestrator.execute_agent(webcrawler_agent, search_query))
                        if search_op_result.get("status") == "success" and search_op_result.get("response_type") == "search_results":
                            search_results = search_op_result.get("response", [])
                            st.success(f"Found {len(search_results)} results for '{search_query}'. Displaying top {num_results_display}.")
                            if search_results:
                                for i, res_item in enumerate(search_results[:num_results_display]): st.markdown(f"**{i+1}. [{res_item.get('title', 'No Title')}]({res_item.get('url')})**"); st.caption(res_item.get('url')); st.markdown(f"> {res_item.get('snippet', 'No snippet available.')}"); st.markdown("---")
                            else: st.info("No search results found.")
                        else: st.error(f"Web search failed: {search_op_result.get('response', 'Unknown error')}"); search_results = []
                    else: st.error("WebCrawler agent is not available or active."); search_results = []
                if search_results:
                    with st.spinner("Analyzing search results..."):
                        analysis_prompt_content = f"Analyze these top {num_results_display} web search results for the query '{search_query}':\n"
                        for i, sr_item in enumerate(search_results[:num_results_display]): analysis_prompt_content += f"\nResult {i+1}:\nTitle: {sr_item.get('title')}\nURL: {sr_item.get('url')}\nSnippet: {sr_item.get('snippet')}\n---"
                        analysis_results = asyncio.run(orchestrator.parallel_execution(analysis_prompt_content, sel_agents)) # Use sel_agents
                        st.subheader("AI Analysis of Search Results:")
                        for res_sa in analysis_results:
                            with st.expander(f"{res_sa.get('agent','N/A')} Analysis"): st.write(str(res_sa.get("response","N/A")))
            else: st.warning("Please enter a search query.")
    elif web_task == "Scrape Single URL & Analyze": # This part was already good
        url_to_scrape = st.text_input("Enter URL to Scrape:", placeholder="https://example.com", key="web_scrape_url_rp")
        if st.button("Scrape URL & Analyze Content", key="web_scrape_button_rp"):
            if url_to_scrape:
                with st.spinner(f"Scraping {url_to_scrape}..."):
                    webcrawler_agent = next((a for a in orchestrator.agents if a.name == "WebCrawler" and a.active), None)
                    if webcrawler_agent:
                        scrape_res = asyncio.run(orchestrator.execute_agent(webcrawler_agent, url_to_scrape))
                        if scrape_res.get("status") == "success": st.success(f"Processed URL: {url_to_scrape}"); st.markdown("**Summary/Response:**"); st.markdown(scrape_res.get("response", "N/A")); # ... (other details)
                        else: st.error(f"Failed to process URL: {scrape_res.get('response', 'Error')}")
                    else: st.error("WebCrawler agent not available.")
            else: st.warning("Please enter a URL.")

def render_image_generation_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("🎨 IMAGE GENERATION STUDIO")
    # ... (all logic from the original "Image Generation" elif block) ...
    image_prompt = st.text_area("Enter your image description:", height=100, placeholder="E.g., 'A photorealistic cat astronaut on the moon'", key="img_gen_prompt_rp")
    if st.button("Generate Image", type="primary", key="img_gen_button_rp"):
        if image_prompt:
            with st.spinner("Generating image..."):
                imageforge_agent = next((a for a in orchestrator.agents if a.name == "ImageForge" and a.active), None)
                if imageforge_agent:
                    result_img = asyncio.run(orchestrator.execute_agent(imageforge_agent, image_prompt)) # temp is not used by ImageForge
                    if result_img.get("status") == "success" and result_img.get("image_path"): st.image(result_img["image_path"], caption=f"Generated for: {image_prompt}"); st.success(f"Image saved to: {result_img['image_path']}")
                    else: st.error(f"Image generation failed: {result_img.get('response', 'Unknown error')}")
                else: st.error("ImageForge agent not available/active.")
        else: st.warning("Please enter an image description.")

def render_video_processing_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("🎞️ VIDEO PROCESSING UTILITIES")
    # ... (all logic from the original "Video Processing" elif block, including temp file cleanup) ...
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"], key="video_uploader_rp")
    if uploaded_video is not None:
        temp_dir = Path(orchestrator.video_processing_dir) / "temp_uploads"; temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video_path = temp_dir / uploaded_video.name
        with open(temp_video_path, "wb") as f: f.write(uploaded_video.getbuffer())
        st.video(str(temp_video_path))
        st.markdown("---"); st.subheader("Get Video Metadata")
        if st.button("Extract Metadata", key="video_meta_button_rp"):
            with st.spinner("Extracting..."):
                try:
                    result_vm = asyncio.run(orchestrator.get_video_metadata(str(temp_video_path)))
                    if result_vm.get("status") == "success": st.json(result_vm.get("metadata")); display_feedback_widgets(f"meta_{uploaded_video.name}", "video_metadata_result", f"meta_{uploaded_video.name}_fb", op_mode)
                    else: st.error(f"Failed: {result_vm.get('message')}")
                finally:
                    if temp_video_path.exists():
                        try: temp_video_path.unlink()
                        except Exception as e: st.warning(f"Cleanup failed: {e}")
        # ... (similar structure for Extract Frame and Convert to GIF, ensuring `finally` block for cleanup)
        st.markdown("---"); st.subheader("Extract Frame")
        extract_ts = st.text_input("Timestamp for frame (e.g., 00:00:05 or 5.0)", key="video_extract_ts_rp")
        if st.button("Extract Frame", key="video_extract_frame_button_rp"):
            if extract_ts:
                with st.spinner("Extracting frame..."):
                    try:
                        result_vf = asyncio.run(orchestrator.extract_video_frame(str(temp_video_path), extract_ts))
                        if result_vf.get("status") == "success" and result_vf.get("frame_path"): st.image(result_vf.get("frame_path"), caption=f"Frame at {extract_ts}"); st.success(f"Frame saved to: {result_vf.get('frame_path')}"); display_feedback_widgets(f"frame_{uploaded_video.name}_{extract_ts}", "video_frame_extraction_result", f"frame_{uploaded_video.name}_{extract_ts}_fb", op_mode)
                        else: st.error(f"Failed to extract frame: {result_vf.get('message')}")
                    finally:
                        if temp_video_path.exists():
                            try: temp_video_path.unlink()
                            except Exception as e: st.warning(f"Cleanup failed for frame: {e}")
            else: st.warning("Please enter a timestamp.")
        st.markdown("---"); st.subheader("Convert to GIF")
        gif_start_ts = st.text_input("GIF Start Timestamp (e.g., 00:00:03 or 3.0)", key="video_gif_start_ts_rp")
        gif_end_ts = st.text_input("GIF End Timestamp (e.g., 00:00:08 or 8.0)", key="video_gif_end_ts_rp")
        if st.button("Convert to GIF", key="video_convert_gif_button_rp"):
            if gif_start_ts and gif_end_ts:
                with st.spinner("Converting to GIF..."):
                    try:
                        result_vg = asyncio.run(orchestrator.convert_video_to_gif(str(temp_video_path), gif_start_ts, gif_end_ts))
                        if result_vg.get("status") == "success" and result_vg.get("gif_path"): st.image(result_vg.get("gif_path"), caption=f"GIF from {gif_start_ts} to {gif_end_ts}"); st.success(f"GIF saved to: {result_vg.get('gif_path')}"); display_feedback_widgets(f"gif_{uploaded_video.name}_{gif_start_ts}_{gif_end_ts}", "video_gif_conversion_result", f"gif_{uploaded_video.name}_{gif_start_ts}_{gif_end_ts}_fb", op_mode)
                        else: st.error(f"Failed to convert to GIF: {result_vg.get('message')}")
                    finally:
                        if temp_video_path.exists():
                            try: temp_video_path.unlink()
                            except Exception as e: st.warning(f"Cleanup failed for GIF: {e}")
            else: st.warning("Please enter both start and end timestamps.")


def render_audio_processing_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("🎤 AUDIO PROCESSING SUITE")
    # ... (all logic from the original "Audio Processing" elif block, including temp file cleanup) ...
    audio_task = st.radio("Select Audio Task:", ("Get Audio Info", "Convert Audio Format", "Text-to-Speech"), key="audio_task_radio_rp")
    if audio_task == "Text-to-Speech":
        st.markdown("---"); tts_text = st.text_area("Text to convert to speech:", key="tts_text_input_rp", height=150)
        tts_filename_stem = st.text_input("Output filename stem (optional):", value="speech_output", key="tts_filename_stem_rp")
        if st.button("Generate Speech", key="tts_generate_button_rp"):
            if tts_text.strip():
                with st.spinner("Generating speech..."):
                    result_tts = asyncio.run(orchestrator.text_to_speech(tts_text, tts_filename_stem))
                    if result_tts.get("status") == "success" and result_tts.get("speech_path"): st.audio(result_tts.get("speech_path"), format='audio/mp3'); st.success(f"Speech saved to: {result_tts.get('speech_path')}"); display_feedback_widgets(f"tts_{tts_filename_stem}", "tts_result", f"tts_{tts_filename_stem}_fb", op_mode, tts_text[:100])
                    else: st.error(f"Failed to generate speech: {result_tts.get('message')}")
            else: st.warning("Please enter text for speech conversion.")
    else:
        st.markdown("---"); uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a"], key="audio_uploader_rp")
        if uploaded_audio is not None:
            temp_audio_dir = Path(orchestrator.audio_processing_dir) / "temp_uploads"; temp_audio_dir.mkdir(parents=True, exist_ok=True)
            temp_audio_path = temp_audio_dir / uploaded_audio.name
            with open(temp_audio_path, "wb") as f: f.write(uploaded_audio.getbuffer())
            st.audio(str(temp_audio_path))
            if audio_task == "Get Audio Info":
                st.markdown("---")
                if st.button("Get Audio Info", key="audio_info_button_rp"):
                    with st.spinner("Extracting audio information..."):
                        try:
                            result_ai = asyncio.run(orchestrator.get_audio_info(str(temp_audio_path)))
                            if result_ai.get("status") == "success": st.json(result_ai.get("info")); display_feedback_widgets(f"info_{uploaded_audio.name}", "audio_info_result", f"info_{uploaded_audio.name}_fb", op_mode)
                            else: st.error(f"Failed to get audio info: {result_ai.get('message')}")
                        finally:
                            if temp_audio_path.exists():
                                try: temp_audio_path.unlink()
                                except Exception as e: st.warning(f"Cleanup failed: {e}")
            elif audio_task == "Convert Audio Format":
                st.markdown("---"); target_format = st.selectbox("Target Format:", ["mp3", "wav", "ogg", "flac"], key="audio_convert_format_select_rp")
                if st.button("Convert Format", key="audio_convert_button_rp"):
                    with st.spinner(f"Converting to {target_format}..."):
                        try:
                            result_ac = asyncio.run(orchestrator.convert_audio_format(str(temp_audio_path), target_format))
                            if result_ac.get("status") == "success" and result_ac.get("output_path"):
                                st.success(f"Converted audio saved to: {result_ac.get('output_path')}")
                                try: st.audio(result_ac.get("output_path"))
                                except Exception as e_ad: st.warning(f"Could not display converted audio: {e_ad}")
                                display_feedback_widgets(f"convert_{uploaded_audio.name}_{target_format}", "audio_conversion_result", f"convert_{uploaded_audio.name}_{target_format}_fb", op_mode)
                            else: st.error(f"Failed to convert audio: {result_ac.get('message')}")
                        finally:
                            if temp_audio_path.exists():
                                try: temp_audio_path.unlink()
                                except Exception as e: st.warning(f"Cleanup failed: {e}")


def render_code_generation_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("💻 PROJECT SCAFFOLDING & CODE GENERATION")
    # ... (all logic from the original "Code Generation" elif block) ...
    code_task = st.selectbox("Select Code Task:", ["Scaffold New Project", "Explain Code Snippet", "Generate Code Module/Class"], key="code_task_select_rp")
    if code_task == "Scaffold New Project":
        st.markdown("---"); st.write("Scaffold a new project structure using AutoDev.")
        project_name = st.text_input("Project Name:", key="scaffold_project_name_rp")
        project_type_options = ["python_cli", "streamlit_dashboard", "nodejs_api"]
        project_type = st.selectbox("Project Type:", project_type_options, key="scaffold_project_type_rp")
        if st.button("Scaffold Project", key="scaffold_project_button_rp"):
            if project_name and project_type:
                with st.spinner(f"Scaffolding '{project_name}'..."):
                    result_sp = asyncio.run(orchestrator.scaffold_new_project(project_name, project_type))
                    if result_sp.get("status") == "success": st.success(result_sp.get("message")); display_feedback_widgets(f"scaffold_{project_name}", "project_scaffolding_result", f"scaffold_{project_name}_fb", op_mode, f"Scaffold: {project_name} ({project_type})")
                    else: st.error(f"Failed to scaffold: {result_sp.get('message')}")
            else: st.warning("Please provide project name and type.")
    elif code_task == "Explain Code Snippet":
        st.markdown("---"); st.write("Get an AI-powered explanation for a code snippet.")
        code_snippet_explain = st.text_area("Code Snippet to Explain:", height=200, key="explain_code_snippet_input_rp")
        if st.button("Explain Snippet", key="explain_snippet_button_rp"):
            if code_snippet_explain.strip():
                with st.spinner("Generating explanation..."):
                    result_ecs = asyncio.run(orchestrator.explain_code_snippet(code_snippet_explain))
                    if result_ecs.get("status") == "success": st.markdown("**Explanation:**"); st.markdown(result_ecs.get("explanation")); display_feedback_widgets(f"explain_{hash(code_snippet_explain)}", "code_explanation_result", f"explain_{hash(code_snippet_explain)}_fb", op_mode, code_snippet_explain[:100])
                    else: st.error(f"Failed to get explanation: {result_ecs.get('message')}")
            else: st.warning("Please enter a code snippet.")
    elif code_task == "Generate Code Module/Class":
        st.markdown("---"); st.write("Generate a new code module or class based on requirements.")
        requirements_generate = st.text_area("Requirements/Description:", height=150, key="generate_module_requirements_rp")
        if st.button("Generate Module", key="generate_module_button_rp"):
            if requirements_generate.strip():
                with st.spinner("Generating code module..."):
                    result_gcm = asyncio.run(orchestrator.generate_code_module(requirements_generate))
                    if result_gcm.get("status") == "success": st.markdown("**Generated Code:**"); st.code(result_gcm.get("generated_code"), language="python"); display_feedback_widgets(f"genmodule_{hash(requirements_generate)}", "code_module_generation_result", f"genmodule_{hash(requirements_generate)}_fb", op_mode, requirements_generate[:100])
                    else: st.error(f"Failed to generate module: {result_gcm.get('message')}")
            else: st.warning("Please provide requirements.")

def render_system_information_ui(op_mode: str, sel_agents: List[str], temp: float):
    st.subheader("📊 SYSTEM INFORMATION DASHBOARD")
    # ... (all logic from the original "System Information" elif block) ...
    sys_admin_agent = next((a for a in orchestrator.agents if a.name == "SystemAdmin" and a.active), None)
    if not sys_admin_agent: st.error("SystemAdmin agent not available."); return
    sys_info_tasks = {"OS Details": "os_info", "CPU Details": "cpu_info", "Disk Space": "disk_space", "Memory Usage": "memory_usage", "Network Config": "network_config"}
    cols = st.columns(len(sys_info_tasks))
    for i, (label, cmd_key) in enumerate(sys_info_tasks.items()):
        if cols[i].button(label, key=f"sysinfo_{cmd_key}_btn_rp"):
            with st.spinner(f"Fetching {label}..."):
                descriptive_prompt = f"get {cmd_key.replace('_', ' ')}"
                result_si = asyncio.run(orchestrator.execute_agent(sys_admin_agent, descriptive_prompt))
                with st.expander(f"{label} Output", expanded=True):
                    if result_si.get("status") == "success":
                        if isinstance(result_si.get("data"), dict): st.json(result_si.get("data"))
                        else: st.text(str(result_si.get("data","N/A")))
                    else: st.error(result_si.get("message", "Failed to fetch info."))
    st.markdown("---"); top_n = st.number_input("Number of processes for 'Top Processes':", 1, 50, 10, key="sysinfo_top_n_input_rp")
    if st.button("Get Top Processes", key="sysinfo_top_proc_btn_rp"):
        with st.spinner(f"Fetching top {top_n} processes..."):
            result_tp = asyncio.run(orchestrator.execute_agent(sys_admin_agent, f"top {top_n} processes"))
            with st.expander(f"Top {top_n} Processes Output", expanded=True):
                if result_tp.get("status") == "success": st.text(str(result_tp.get("data","N/A")))
                else: st.error(result_tp.get("message", "Failed to fetch processes."))
    st.markdown("---"); st.subheader("Feedback Analysis")
    if st.button("📊 Generate & Store Feedback Analysis Report", key="gen_feedback_report_button_ui_rp"):
        with st.spinner("Generating feedback report..."):
            report_result = asyncio.run(orchestrator.generate_and_store_feedback_report())
            if report_result.get("status") == "success": st.success(report_result.get("message", "Report generated!"));
                if report_result.get("kb_id"): st.caption(f"Stored in KB with ID: {report_result.get('kb_id')}")
            else: st.error(report_result.get("message", "Failed to generate report."))


def main():
   if orchestrator:
        video_temp_base = Path(orchestrator.video_processing_dir)
        cleanup_temp_uploads(video_temp_base, "temp_uploads")
        audio_temp_base = Path(orchestrator.audio_processing_dir)
        cleanup_temp_uploads(audio_temp_base, "temp_uploads")

   st.markdown("""<div style='text-align:center;background:linear-gradient(90deg,#FF6B6B,#4ECDC4,#45B7D1,#96CEB4);padding:20px;border-radius:10px;margin-bottom:20px'>
   <h1 style='color:white;text-shadow:2px 2px 4px rgba(0,0,0,0.5)'>TERMINUS AI NEXUS</h1>
   <p style='color:white;font-size:18px'>ULTIMATE LOCAL AI ECOSYSTEM | ADVANCED ORCHESTRATION</p></div>""",unsafe_allow_html=True)

   with st.sidebar:
       st.header("COMMAND CENTER")
       operation_modes = [
           "Multi-Agent Chat", "Knowledge Base Explorer", "Document Processing",
           "Web Intelligence", "Image Generation", "Video Processing",
           "Audio Processing", "Code Generation", "System Information",
       ]
       operation_mode = st.selectbox("Operation Mode", operation_modes, key="main_operation_mode_sb") # Unique key
       st.subheader("Agent Selection")
       selected_agents_names = [] # Default
       if hasattr(orchestrator, 'agents') and orchestrator.agents:
           agent_names = [a.name for a in orchestrator.agents if a.active]
           default_selection = agent_names[:min(len(agent_names), 3)]
           selected_agents_names = st.multiselect("Active Agents for Parallel Execution", options=agent_names, default=default_selection, key="sidebar_selected_agents_ms") # Unique key
       else:
           st.warning("Orchestrator agents not loaded.")
       st.subheader("LLM Parameters (General)")
       temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, key="llm_temp_sl") # Unique key

   # UI Dispatcher
   ui_render_functions = {
        "Multi-Agent Chat": render_multi_agent_chat_ui,
        "Knowledge Base Explorer": render_knowledge_base_explorer_ui,
        "Document Processing": render_document_processing_ui,
        "Web Intelligence": render_web_intelligence_ui,
        "Image Generation": render_image_generation_ui,
        "Video Processing": render_video_processing_ui,
        "Audio Processing": render_audio_processing_ui,
        "Code Generation": render_code_generation_ui,
        "System Information": render_system_information_ui,
    }

   if operation_mode in ui_render_functions:
       ui_render_functions[operation_mode](operation_mode, selected_agents_names, temperature)
   else:
       st.error(f"Selected operation mode '{operation_mode}' does not have a UI renderer configured.")


if __name__=="__main__":
   try:
       loop = asyncio.get_running_loop()
   except RuntimeError:
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
   if 'orchestrator' in globals() and orchestrator is not None:
       main()
   else:
       print("UI cannot start because core orchestrator components failed to load.")
