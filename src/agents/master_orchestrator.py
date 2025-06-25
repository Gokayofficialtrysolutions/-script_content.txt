import asyncio, json, requests, subprocess, threading, queue, time, datetime
import torch
import aiohttp
from diffusers import DiffusionPipeline
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import pyttsx3
import shutil
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from dataclasses import dataclass
from typing import List,Dict,Any,Optional, Callable, Coroutine
from pathlib import Path
from transformers import pipeline as hf_pipeline
import sys
import re
import platform

try:
    from tools.auto_dev import auto_dev
except ImportError:
    try:
        print("Attempting standard import for tools.auto_dev")
        from tools.auto_dev import auto_dev
    except ImportError as e_imp:
        print(f"CRITICAL: Failed to import auto_dev from tools.auto_dev: {e_imp}")
        print("Ensure that the $INSTALL_DIR is in your PYTHONPATH or launch_terminus.py correctly sets up sys.path.")
        auto_dev = None

import chromadb
from chromadb.utils import embedding_functions
import uuid
from collections import defaultdict

@dataclass
class Agent:
   name:str
   model:str
   specialty:str
   active:bool=True
   estimated_complexity:Optional[str]=None # E.g., "low", "medium", "high"

class TerminusOrchestrator:
   def __init__(self):
       self.agents = []
       self.install_dir = Path(__file__).resolve().parent.parent

       self.agents_config_path = self.install_dir / "agents.json"
       self.models_config_path = self.install_dir / "models.conf"

       self.data_dir = self.install_dir / "data"
       self.logs_dir = self.install_dir / "logs"
       self.tools_dir = self.install_dir / "tools"

       self.generated_images_dir = self.data_dir / "generated_images"
       self.video_processing_dir = self.data_dir / "video_outputs"
       self.audio_processing_dir = self.data_dir / "audio_outputs"
       self.chroma_db_path = str(self.data_dir / "vector_store")
       self.feedback_log_file_path = self.logs_dir / "feedback_log.jsonl"
       self.feedback_analyzer_script_path = self.tools_dir / "feedback_analyzer.py"

       for dir_path in [self.data_dir, self.logs_dir, self.tools_dir,
                        self.generated_images_dir, self.video_processing_dir,
                        self.audio_processing_dir, Path(self.chroma_db_path).parent]:
           dir_path.mkdir(parents=True, exist_ok=True)

       try:
           with open(self.agents_config_path, 'r') as f:
               agents_data = json.load(f)
           for agent_config in agents_data:
               self.agents.append(Agent(**agent_config))
       except Exception as e:
           print(f"ERROR loading agents.json: {e}. No agents loaded.")

       self.ollama_url="http://localhost:11434/api/generate"
       self.image_gen_pipeline = None
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       self.image_gen_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

       try:
           self.tts_engine = pyttsx3.init()
       except Exception as e:
           print(f"WARNING: Failed to initialize TTS engine: {e}.")
           self.tts_engine = None

       self.intent_classifier_model_name = "facebook/bart-large-mnli"
       self.ner_model_name = "dslim/bert-base-NER"
       self.intent_classifier = None
       self.ner_pipeline = None
       self.candidate_intent_labels = [
           "image_generation", "code_generation", "code_modification", "code_explanation",
           "project_scaffolding", "video_info", "video_frame_extraction", "video_to_gif",
           "audio_info", "audio_format_conversion", "text_to_speech",
           "data_analysis", "web_search", "document_processing", "general_question_answering",
           "complex_task_planning", "system_information_query", "knowledge_base_query",
           "feedback_submission", "feedback_analysis_request"
       ]
       try:
           self.intent_classifier = hf_pipeline("zero-shot-classification", model=self.intent_classifier_model_name, device=self.device)
           print("Intent Classifier initialized.")
       except Exception as e: print(f"WARNING: Failed to initialize Intent Classifier: {e}.")
       try:
           self.ner_pipeline = hf_pipeline("ner", model=self.ner_model_name, tokenizer=self.ner_model_name, device=self.device, aggregation_strategy="simple")
           print("NER Pipeline initialized.")
       except Exception as e: print(f"WARNING: Failed to initialize NER Pipeline: {e}.")

       self.conversation_history = []
       self.max_history_items = 10

       self.kb_collection_name = "terminus_knowledge_v1"
       self.knowledge_collection = None
       try:
           self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
           default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
           self.knowledge_collection = self.chroma_client.get_or_create_collection(name=self.kb_collection_name, embedding_function=default_ef)
           print(f"KB initialized: Collection '{self.kb_collection_name}' at {self.chroma_db_path}.")
       except Exception as e: print(f"CRITICAL ERROR initializing ChromaDB: {e}. KB unavailable.")

       self.message_bus_subscribers = defaultdict(list)
       self.message_processing_tasks = set()
       self._setup_initial_event_listeners()
       print("TerminusOrchestrator initialized.")

   def get_agent_capabilities_description(self) -> str:
       descriptions = []
       for a in self.agents:
           if a.active:
               complexity_info = f" (Complexity: {a.estimated_complexity})" if a.estimated_complexity else ""
               descriptions.append(f"- {a.name}: Specializes in '{a.specialty}'. Uses model: {a.model}.{complexity_info}")
       return "\n".join(descriptions) if descriptions else "No active agents available."

   async def _handle_system_event(self, message: Dict):
       print(f"[EVENT_HANDLER] Msg ID: {message.get('message_id')}, Type: '{message.get('message_type')}', Src: '{message.get('source_agent_name')}', Payload: {message.get('payload')}")

   def _setup_initial_event_listeners(self):
       kb_event_types = ["kb.webcontent.added", "kb.code_explanation.added", "kb.code_module.added", "kb.plan_execution_log.added", "kb.document_excerpt.added", "kb.feedback_report.added"]
       for event_type in kb_event_types:
           self.subscribe_to_message(event_type, self._handle_system_event)
           if event_type != "kb.feedback_report.added":
                self.subscribe_to_message(event_type, self._handle_new_kb_content_for_analysis)
       self.subscribe_to_message("user.feedback.submitted", self._handle_system_event)

   async def _handle_new_kb_content_for_analysis(self, message: Dict):
       kb_id = message.get("payload", {}).get("kb_id", "UNKNOWN_KB_ID")
       handler_id = f"[ContentAnalysisHandler kb_id:{kb_id}]"
       print(f"{handler_id} START: Processing message type: {message.get('message_type')}")

       if self.knowledge_collection is None:
           print(f"{handler_id} ERROR: Knowledge base not available. Skipping analysis.")
           return
       if not kb_id or kb_id == "UNKNOWN_KB_ID":
           print(f"{handler_id} ERROR: No valid kb_id in message payload. Cannot process.")
           return
       try:
           item_data = self.knowledge_collection.get(ids=[kb_id], include=["documents"])
           if not (item_data and item_data.get('ids') and item_data['ids'][0]):
               print(f"{handler_id} ERROR: KB item not found for analysis.")
               return
           document_content = item_data['documents'][0]
           if not document_content:
               print(f"{handler_id} INFO: KB item has empty content. Skipping analysis.")
               return

           analysis_agent = next((a for a in self.agents if a.name == "ContentAnalysisAgent" and a.active), None)
           if not analysis_agent:
               print(f"{handler_id} ERROR: ContentAnalysisAgent not found/active. Skipping.")
               return

           analysis_prompt = (
               f"Analyze the following text content:\n---\n{document_content[:15000]}\n---\n"
               f"Provide output as a JSON object with 'keywords' (comma-separated string, or 'NONE') "
               f"and 'topics' (1-3 comma-separated strings, or 'NONE').\n"
               f"Example: {{\"keywords\": \"k1, k2\", \"topics\": \"T1, T2\"}}\nJSON Output:"
           )
           print(f"{handler_id} INFO: Calling LLM for analysis.")
           llm_result = await self.execute_agent(analysis_agent, analysis_prompt)

           if not (llm_result.get("status") == "success" and llm_result.get("response","").strip()):
               print(f"{handler_id} ERROR: LLM analysis call failed. Status: {llm_result.get('status')}, Resp: {llm_result.get('response')}")
               return

           print(f"{handler_id} SUCCESS: LLM analysis successful.")
           llm_response_str = llm_result.get("response").strip()
           extracted_keywords, extracted_topics = "", ""
           try:
               data = json.loads(llm_response_str)
               raw_kw = data.get("keywords","").strip(); extracted_keywords = raw_kw if raw_kw.upper() != "NONE" else ""
               raw_tp = data.get("topics","").strip(); extracted_topics = raw_tp if raw_tp.upper() != "NONE" else ""
           except json.JSONDecodeError:
               print(f"{handler_id} WARNING: Failed to parse LLM JSON. Raw: '{llm_response_str}'. Using raw as keywords if applicable.")
               if "keywords" not in llm_response_str.lower() and "topics" not in llm_response_str.lower() and llm_response_str.upper() != "NONE":
                   extracted_keywords = llm_response_str

           if extracted_keywords or extracted_topics:
               new_meta = {"analysis_by_agent": analysis_agent.name, "analysis_model_used": analysis_agent.model, "analysis_timestamp_iso": datetime.datetime.now().isoformat()}
               if extracted_keywords: new_meta["extracted_keywords"] = extracted_keywords
               if extracted_topics: new_meta["extracted_topics"] = extracted_topics

               print(f"{handler_id} INFO: Attempting metadata update with keywords: '{extracted_keywords}', topics: '{extracted_topics}'.")
               update_status = await self._update_kb_item_metadata(kb_id, new_meta)
               if update_status.get("status") == "success": print(f"{handler_id} SUCCESS: Metadata update successful.")
               else: print(f"{handler_id} ERROR: Metadata update failed. Msg: {update_status.get('message')}")
           else:
               print(f"{handler_id} INFO: No keywords or topics extracted. No metadata update.")
       except Exception as e:
           print(f"{handler_id} UNHANDLED ERROR: {e}")
       finally:
           print(f"{handler_id} END: Finished processing.")

   async def publish_message(self, message_type: str, source_agent_name: str, payload: Dict) -> str:
       message_id = str(uuid.uuid4())
       message = { "message_id": message_id, "message_type": message_type, "source_agent_name": source_agent_name, "timestamp_iso": datetime.datetime.now().isoformat(), "payload": payload }
       print(f"[MessageBus] Publishing: ID={message_id}, Type='{message_type}', Src='{source_agent_name}'")
       for handler in list(self.message_bus_subscribers.get(message_type, [])):
            try:
                if asyncio.iscoroutinefunction(handler): asyncio.create_task(handler(message)).add_done_callback(self.message_processing_tasks.discard)
                elif isinstance(handler, asyncio.Queue): await handler.put(message)
            except Exception as e: print(f"ERROR dispatching message {message_id} to {handler}: {e}")
       return message_id

   def subscribe_to_message(self, message_type: str, handler: Callable[..., Coroutine[Any, Any, None]] | asyncio.Queue):
       self.message_bus_subscribers[message_type].append(handler)
       print(f"[MessageBus] Subscribed '{getattr(handler, '__name__', str(type(handler)))}' to '{message_type}'.")

   async def _execute_single_plan_step(self, step_definition: Dict, full_plan_list: List[Dict], current_step_outputs: Dict) -> Dict:
       step_id = step_definition.get("step_id"); agent_name = step_definition.get("agent_name")
       task_prompt = step_definition.get("task_prompt", ""); dependencies = step_definition.get("dependencies", [])
       output_var_name = step_definition.get("output_variable_name")
       max_retries = step_definition.get("max_retries", 0); retry_delay_seconds = step_definition.get("retry_delay_seconds", 5)
       retry_on_statuses = step_definition.get("retry_on_statuses", ["error"])
       current_execution_retries = 0
       target_agent = next((a for a in self.agents if a.name == agent_name and a.active), None)
       if not target_agent: return {"status": "error", "agent": agent_name, "step_id": step_id, "response": f"Agent '{agent_name}' not found/active."}
       while True:
           current_task_prompt = task_prompt
           for dep_id in dependencies:
               dep_key = next((p.get("output_variable_name",f"step_{dep_id}_output") for p in full_plan_list if p.get("step_id")==dep_id),None)
               if dep_key and dep_key in current_step_outputs:
                   val = current_step_outputs[dep_key]
                   if isinstance(val,dict):
                       for m in re.finditer(r"{{{{("+re.escape(dep_key)+r")\.(\w+)}}}}",current_task_prompt):
                           if m.group(2) in val: current_task_prompt=current_task_prompt.replace(m.group(0),str(val[m.group(2)]))
                   current_task_prompt=current_task_prompt.replace(f"{{{{{dep_key}}}}}",str(val))
           log_msg = f"Executing step {step_id}" + (f" (Retry {current_execution_retries}/{max_retries})" if current_execution_retries > 0 else "")
           print(f"{log_msg}: Agent='{agent_name}', Prompt='{current_task_prompt[:100]}...'")
           step_result = await self.execute_agent(target_agent, current_task_prompt)
           if step_result.get("status") == "success":
               key_to_store = output_var_name if output_var_name else f"step_{step_id}_output"
               current_step_outputs[key_to_store] = step_result.get("response")
               for mk in ["image_path","frame_path","gif_path","speech_path","modified_file"]:
                   if mk in step_result: current_step_outputs[f"{key_to_store}_{mk}"]=step_result[mk]
               return step_result
           current_execution_retries+=1
           if not(current_execution_retries <= max_retries and step_result.get("status") in retry_on_statuses): return step_result
           await asyncio.sleep(retry_delay_seconds)

   async def store_knowledge(self, content: str, metadata: Optional[Dict] = None, content_id: Optional[str] = None) -> Dict:
       if self.knowledge_collection is None: return {"status": "error", "message": "KB not initialized."}
       try:
           final_id = content_id or str(uuid.uuid4())
           clean_meta = {k: (str(v) if not isinstance(v, (str,int,float,bool)) else v) for k,v in metadata.items()} if metadata else {}
           self.knowledge_collection.add(ids=[final_id], documents=[content], metadatas=[clean_meta] if clean_meta else [None])
           return {"status": "success", "id": final_id, "message": "Content stored."}
       except Exception as e: return {"status": "error", "message": str(e)}

   async def retrieve_knowledge(self, query_text: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> Dict:
        if self.knowledge_collection is None: return {"status": "error", "results": [], "message": "KB not initialized."}
        try:
            clean_filter = {k:v for k,v in filter_metadata.items() if isinstance(v,(str,int,float,bool))} if filter_metadata else None
            q_res = self.knowledge_collection.query(query_texts=[query_text],n_results=max(1,n_results),where=clean_filter)
            results = []
            if q_res and q_res.get('ids') and q_res['ids'][0]:
                for i, item_id in enumerate(q_res['ids'][0]):
                    results.append({"id":item_id, "document":q_res['documents'][0][i], "metadata":q_res['metadatas'][0][i], "distance":q_res['distances'][0][i]})
            return {"status":"success", "results":results}
        except Exception as e: return {"status":"error", "results":[], "message":str(e)}

   def store_user_feedback(self, item_id: str, item_type: str, rating: str,
                           comment: Optional[str] = None, current_mode: Optional[str] = None,
                           user_prompt_preview: Optional[str] = None) -> bool:
       try:
           data = {"feedback_id":str(uuid.uuid4()), "timestamp_iso":datetime.datetime.now().isoformat(), "item_id":str(item_id), "item_type":str(item_type), "rating":str(rating), "comment":comment or "", "user_context":{"operation_mode":current_mode, "related_user_prompt_preview":user_prompt_preview[:200] if user_prompt_preview else None}}
           with open(self.feedback_log_file_path, 'a', encoding='utf-8') as f: f.write(json.dumps(data) + '\n')
           asyncio.create_task(self.publish_message("user.feedback.submitted", "UserFeedbackSystem", {"feedback_id":data["feedback_id"]}))
           return True
       except Exception as e: print(f"ERROR storing feedback: {e}"); return False

   async def generate_and_store_feedback_report(self) -> Dict:
       report_handler_id = "[FeedbackReport]"
       print(f"{report_handler_id} START: Generating and storing feedback analysis report.")
       if self.knowledge_collection is None: return {"status": "error", "message": "KB not initialized."}
       if not self.feedback_analyzer_script_path.exists(): return {"status": "error", "message": f"Analyzer script missing: {self.feedback_analyzer_script_path}"}
       try:
           print(f"{report_handler_id} INFO: Executing {self.feedback_analyzer_script_path} with log {self.feedback_log_file_path}")
           proc = await asyncio.create_subprocess_exec(sys.executable, str(self.feedback_analyzer_script_path), f"--log_file={self.feedback_log_file_path}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE) # Use PIPE from asyncio.subprocess
           stdout, stderr = await proc.communicate()

           if proc.returncode != 0:
               err_msg = stderr.decode().strip() if stderr else "Unknown error from feedback_analyzer.py"
               print(f"{report_handler_id} ERROR: Analyzer script failed. Code: {proc.returncode}. Err: {err_msg}")
               return {"status": "error", "message": f"Analyzer script failed: {err_msg}"}
           print(f"{report_handler_id} SUCCESS: Analyzer script executed.")

           report_json_str = stdout.decode().strip()
           if not report_json_str: print(f"{report_handler_id} WARNING: Analyzer script produced no output."); return {"status": "error", "message":"Analyzer script empty output."}

           try: report_data = json.loads(report_json_str)
           except json.JSONDecodeError as e: print(f"{report_handler_id} ERROR: Failed to parse JSON from analyzer: {e}. Output: {report_json_str[:200]}..."); return {"status":"error", "message":f"Bad JSON from analyzer: {e}"}
           print(f"{report_handler_id} SUCCESS: Parsed report JSON.")

           expected_keys = ["report_id", "report_generation_timestamp_iso", "analysis_period_start_iso", "analysis_period_end_iso", "total_feedback_entries_processed", "overall_sentiment_distribution", "sentiment_by_item_type", "comment_previews"]
           missing_keys = [k for k in expected_keys if k not in report_data]
           if missing_keys: print(f"{report_handler_id} WARNING: Report JSON missing keys: {missing_keys}.")
           if "overall_sentiment_distribution" in report_data and not isinstance(report_data["overall_sentiment_distribution"],dict): print(f"{report_handler_id} WARNING: 'overall_sentiment_distribution' not a dict.")
           if "total_feedback_entries_processed" in report_data and not isinstance(report_data["total_feedback_entries_processed"],int): print(f"{report_handler_id} WARNING: 'total_feedback_entries_processed' not an int.")

           kb_meta = {"source":"feedback_analysis_report", "report_id":report_data.get("report_id",str(uuid.uuid4())), "report_date_iso":report_data.get("report_generation_timestamp_iso",datetime.datetime.now().isoformat()).split('T')[0], "analysis_start_iso":report_data.get("analysis_period_start_iso"), "analysis_end_iso":report_data.get("analysis_period_end_iso")}
           kb_meta = {k:v for k,v in kb_meta.items() if v is not None}

           print(f"{report_handler_id} INFO: Storing report in KB. Report ID: {kb_meta.get('report_id')}")
           store_res = await self.store_knowledge(content=report_json_str, metadata=kb_meta) # Use report_json_str

           if store_res.get("status") == "success":
               msg = f"Feedback report stored. KB ID: {store_res.get('id')}"
               print(f"{report_handler_id} SUCCESS: {msg}")
               asyncio.create_task(self.publish_message("kb.feedback_report.added","FeedbackAnalyzerSystem",{"report_id":kb_meta.get('report_id'),"kb_id":store_res.get("id")}))
               return {"status":"success", "message":msg, "kb_id":store_res.get("id")}
           else: print(f"{report_handler_id} ERROR: Failed to store report in KB. Msg: {store_res.get('message')}"); return store_res
       except Exception as e: print(f"{report_handler_id} UNHANDLED ERROR: {e}"); return {"status":"error","message":str(e)}
       finally: print(f"{report_handler_id} END: Processing finished.")

   async def _update_kb_item_metadata(self, kb_id: str, new_metadata_fields: Dict) -> Dict:
       if self.knowledge_collection is None: return {"status": "error", "message": "KB not initialized."}
       try:
           existing = self.knowledge_collection.get(ids=[kb_id], include=["metadatas","documents"])
           if not (existing and existing.get('ids') and existing['ids'][0]): return {"status":"error", "message":f"KB ID '{kb_id}' not found."}
           current_meta = existing['metadatas'][0] or {}; doc = existing['documents'][0]
           if doc is None: return {"status":"error", "message":f"Doc for KB ID '{kb_id}' missing."}
           updated_meta = current_meta.copy()
           for k,v in new_metadata_fields.items(): updated_meta[k] = str(v) if not isinstance(v,(str,int,float,bool)) else v
           self.knowledge_collection.update(ids=[kb_id],metadatas=[updated_meta],documents=[doc])
           return {"status":"success", "id":kb_id, "message":"Metadata updated."}
       except Exception as e: return {"status":"error", "message":str(e)}

   def get_conversation_history_for_display(self) -> List[Dict]:
       return list(self.conversation_history)

   async def execute_master_plan(self, user_prompt: str) -> List[Dict]:
       plan_handler_id = f"[MasterPlanner user_prompt:'{user_prompt[:50]}...']"
       print(f"{plan_handler_id} START: Received request.")
       self.conversation_history.append({"role": "user", "content": user_prompt})
       if len(self.conversation_history) > self.max_history_items: self.conversation_history = self.conversation_history[-self.max_history_items:]

       max_rev_attempts = 1; current_attempt = 0; original_plan_json_str = ""; final_exec_results = []
       first_attempt_nlu_output = {}; kb_general_ctx_str = ""; kb_plan_log_ctx_str = ""; kb_feedback_ctx_str = ""
       detailed_failure_ctx_for_rev = {}
       plan_succeeded_this_attempt = False # Initialize

       print(f"{plan_handler_id} INFO: Performing NLU analysis.")
       first_attempt_nlu_output = await self.classify_user_intent(user_prompt)
       nlu_summary_for_prompt = f"NLU Analysis :: Intent: {first_attempt_nlu_output.get('intent','N/A')} :: Entities: {str(first_attempt_nlu_output.get('entities',[]))[:100]}..."
       print(f"{plan_handler_id} INFO: {nlu_summary_for_prompt}")

       while current_attempt <= max_rev_attempts:
           current_attempt += 1; current_plan_json_str = ""
           print(f"{plan_handler_id} INFO: Attempt {current_attempt}/{max_rev_attempts + 1}")

           if current_attempt == 1:
               if self.knowledge_collection:
                   query_text_general = user_prompt + " " + " ".join([e.get('text',"") for e in first_attempt_nlu_output.get('entities',[])])
                   kb_res_general = await self.retrieve_knowledge(query_text_general, n_results=2)
                   if kb_res_general.get("status")=="success" and kb_res_general.get("results"):
                       entries = [self._format_kb_entry_for_prompt(item) for item in kb_res_general["results"]]
                       if entries: kb_general_ctx_str = "General KB Context:\n" + "\n".join(entries)
               if self.knowledge_collection and first_attempt_nlu_output.get("intent"):
                   query_text_plan_logs = user_prompt + " " + first_attempt_nlu_output.get('intent','N/A')
                   kb_res_logs = await self.retrieve_knowledge(query_text_plan_logs, n_results=1, filter_metadata={"source":"plan_execution_log", "primary_intent":first_attempt_nlu_output.get("intent")})
                   if kb_res_logs.get("status")=="success" and kb_res_logs.get("results"):
                       kb_plan_log_ctx_str = self._format_plan_log_entry_for_prompt(kb_res_logs["results"][0])
               if self.knowledge_collection:
                   query_text_feedback = f"masterplanner feedback report intent {first_attempt_nlu_output.get('intent','general')}"
                   kb_res_feedback = await self.retrieve_knowledge(query_text_feedback, n_results=1, filter_metadata={"source":"feedback_analysis_report"})
                   if kb_res_feedback.get("status")=="success" and kb_res_feedback.get("results"):
                       kb_feedback_ctx_str = self._format_feedback_report_for_prompt(kb_res_feedback["results"][0])

           current_planner_prompt = ""
           if current_attempt == 1:
                current_planner_prompt = self._construct_main_planning_prompt(
                    user_prompt,
                    self._get_relevant_history_for_prompt(user_prompt),
                    nlu_summary_for_prompt,
                    kb_general_ctx_str,
                    kb_plan_log_ctx_str,
                    kb_feedback_ctx_str,
                    self.get_agent_capabilities_description()
                )
           else: # Revision attempt
                current_planner_prompt = self._construct_revision_planning_prompt(
                    user_prompt,
                    self._get_relevant_history_for_prompt(user_prompt, full_history=True),
                    nlu_summary_for_prompt,
                    detailed_failure_ctx_for_rev,
                    self.get_agent_capabilities_description()
                )

           print(f"{plan_handler_id} INFO: Prompting MasterPlanner LLM (Attempt {current_attempt}).")
           llm_response = await self.execute_agent(next(a for a in self.agents if a.name=="MasterPlanner"), current_planner_prompt)
           current_plan_json_str = llm_response.get("response","").strip() if llm_response.get("status")=="success" else ""
           if current_attempt == 1: original_plan_json_str = current_plan_json_str

           if not current_plan_json_str: final_exec_results = [{"status":"error","response":"MasterPlanner returned empty plan."}]; break
           try: plan_list = json.loads(current_plan_json_str)
           except Exception as e: final_exec_results=[{"status":"error","response":f"Plan JSON parsing error: {e}"}]; break
           if not isinstance(plan_list, list): final_exec_results=[{"status":"error","response":"Plan is not a list."}]; break
           if not plan_list and current_attempt == 1: final_exec_results=[{"status":"info","response":"MasterPlanner returned empty list (simple task or unplannable)."}]; break
           if not plan_list and current_attempt > 1: final_exec_results=[{"status":"error","response":"MasterPlanner revision returned empty list."}]; break

           step_outputs = {}
           executed_step_ids = set()
           current_step_idx = 0 # Use index for plan_list iteration
           plan_steps_map = {step.get("step_id"): step for step in plan_list}
           current_attempt_results = []
           plan_succeeded_this_attempt = True

           while current_step_idx < len(plan_list):
               # This assumes plan_list is sorted or step_ids allow sequential lookup by index initially.
               # If a conditional jump changes current_step_idx, the next iteration uses that new index.
               step_to_execute = plan_list[current_step_idx]
               step_id_to_execute = step_to_execute.get("step_id")

               if step_id_to_execute in executed_step_ids:
                   print(f"{plan_handler_id} WARNING: Step {step_id_to_execute} already executed in this attempt. Skipping to prevent loop.")
                   current_step_idx += 1
                   continue

               # Basic dependency check (more robust needed for complex graphs)
               deps_met = True
               for dep_id in step_to_execute.get("dependencies", []):
                   if dep_id not in executed_step_ids:
                       print(f"{plan_handler_id} ERROR: Dependency {dep_id} for step {step_id_to_execute} not met. Halting plan attempt.")
                       current_attempt_results.append({"step_id": step_id_to_execute, "status": "error", "response": f"Dependency {dep_id} not met."})
                       plan_succeeded_this_attempt = False; break
               if not plan_succeeded_this_attempt: break

               step_type = step_to_execute.get("step_type")
               step_result_for_this_iteration = None

               if step_type == "conditional":
                   condition_eval_result = await self._evaluate_plan_condition(step_to_execute.get("condition", {}), step_outputs, plan_list)
                   step_result_for_this_iteration = {
                       "step_id": step_id_to_execute, "step_type": "conditional", "status": condition_eval_result.get("status"),
                       "condition_evaluation": condition_eval_result.get("evaluation"), "message": condition_eval_result.get("message") }
                   current_attempt_results.append(step_result_for_this_iteration)
                   executed_step_ids.add(step_id_to_execute)

                   if condition_eval_result.get("status") == "success":
                       target_step_id = step_to_execute.get("if_true_step_id") if condition_eval_result.get("evaluation") else step_to_execute.get("if_false_step_id")
                       if target_step_id and target_step_id in plan_steps_map:
                           try: current_step_idx = plan_list.index(plan_steps_map[target_step_id]); continue
                           except ValueError: # Should not happen if in plan_steps_map
                               plan_succeeded_this_attempt = False; print(f"Jump target {target_step_id} in map but not list index."); break
                       elif target_step_id: # Defined but not in plan
                           plan_succeeded_this_attempt = False; print(f"Jump target {target_step_id} not in plan."); break
                       else: current_step_idx +=1 # No jump, proceed sequentially
                   else: # Condition evaluation failed
                       plan_succeeded_this_attempt = False; break

               elif step_to_execute.get("agent_name") == "parallel_group":
                   # This is a simplified placeholder. Real parallel execution is more complex.
                   print(f"{plan_handler_id} INFO: Executing parallel group {step_id_to_execute} (mocked).")
                   step_result_for_this_iteration = {"step_id": step_id_to_execute, "agent": "parallel_group", "status": "success", "response": "Mock parallel success"}
                   current_attempt_results.append(step_result_for_this_iteration)
                   step_outputs[step_id_to_execute] = step_result_for_this_iteration.get("response")
                   executed_step_ids.add(step_id_to_execute)
                   current_step_idx += 1
               else: # Regular agent step
                   step_result_for_this_iteration = await self._execute_single_plan_step(step_to_execute, plan_list, step_outputs)
                   current_attempt_results.append(step_result_for_this_iteration)
                   if step_result_for_this_iteration.get("status") != "success":
                       plan_succeeded_this_attempt = False; break
                   executed_step_ids.add(step_id_to_execute)
                   current_step_idx += 1

           final_exec_results = current_attempt_results # Store results of this attempt
           if plan_succeeded_this_attempt: print(f"{plan_handler_id} SUCCESS: Plan attempt {current_attempt} succeeded."); break

           # If plan failed, capture context for revision
           if not plan_succeeded_this_attempt and current_attempt_results:
                last_failed_step_info = current_attempt_results[-1]
                failed_step_def_for_rev = plan_steps_map.get(last_failed_step_info.get("step_id")) if last_failed_step_info.get("step_id") else step_to_execute # approx
                detailed_failure_ctx_for_rev = self._capture_failure_context(
                    current_plan_json_str, # Use the plan string from this failed attempt
                    failed_step_def_for_rev,
                    last_failed_step_info,
                    step_outputs
                )
           if current_attempt >= max_rev_attempts: print(f"{plan_handler_id} ERROR: Max revisions reached."); break

       assistant_summary = f"Plan executed. Overall status: {'Success' if plan_succeeded_this_attempt else 'Failed'}. {len(final_exec_results)} steps processed in final attempt."
       plan_log_kb_id = None
       if self.knowledge_collection:
            log_content = json.dumps({"request":user_prompt, "nlu":first_attempt_nlu_output, "plan_used_for_final_attempt":current_plan_json_str, "final_outcome_status": plan_succeeded_this_attempt, "final_attempt_step_results":final_exec_results, "summary":assistant_summary})
            kb_log_res = await self.store_knowledge(content=log_content, metadata={"source":"plan_execution_log", "primary_intent":first_attempt_nlu_output.get('intent')})
            if kb_log_res.get("status") == "success": plan_log_kb_id = kb_log_res.get("id")

       self.conversation_history.append({"role":"assistant", "content":assistant_summary, "plan_log_kb_id": plan_log_kb_id})
       print(f"{plan_handler_id} END: Execution complete. Summary: {assistant_summary}")
       return final_exec_results

   async def _evaluate_plan_condition(self, condition_def: Dict, step_outputs: Dict, full_plan_list: List[Dict]) -> Dict:
       """
       Evaluates a condition defined in a conditional plan step.
       Returns: Dict with "status" ("success" or "error"), "evaluation" (True/False/None), "message".
       """
       if not condition_def:
           return {"status": "error", "evaluation": None, "message": "Condition definition is empty."}

       source_step_id = condition_def.get("source_step_id")
       output_var_path = condition_def.get("source_output_variable") # Can be dot-separated for nested access
       operator = condition_def.get("operator")
       compare_value_literal = condition_def.get("value")
       value_type_hint = condition_def.get("value_type", "string") # Default to string comparison

       if not all([source_step_id, output_var_path, operator]):
           return {"status": "error", "evaluation": None, "message": "Condition missing source_step_id, source_output_variable, or operator."}

       source_step_output_key = None
       for step_cfg in full_plan_list: # Iterate through the original plan list to find the source step's output name
           if step_cfg.get("step_id") == source_step_id:
               source_step_output_key = step_cfg.get("output_variable_name", f"step_{source_step_id}_output")
               break

       if not source_step_output_key or source_step_output_key not in step_outputs:
           return {"status": "error", "evaluation": None, "message": f"Output for source_step_id '{source_step_id}' (expected key: {source_step_output_key}) not found in step_outputs: {list(step_outputs.keys())}."}

       actual_value_raw = step_outputs.get(source_step_output_key)

       try:
           current_val = actual_value_raw
           for part in output_var_path.split('.'):
               if isinstance(current_val, dict):
                   current_val = current_val.get(part)
                   if current_val is None: break
               elif isinstance(current_val, list) and part.isdigit() and int(part) < len(current_val): # Basic list index access
                    current_val = current_val[int(part)]
               else: current_val = None; break
           actual_value_to_compare = current_val
       except Exception as e:
           return {"status": "error", "evaluation": None, "message": f"Error accessing source_output_variable '{output_var_path}' from '{source_step_output_key}': {str(e)}"}

       try:
           if value_type_hint == "integer":
               actual_value_to_compare = int(actual_value_to_compare) if actual_value_to_compare is not None else None
               compare_value_literal = int(compare_value_literal) if compare_value_literal is not None else None
           elif value_type_hint == "float":
               actual_value_to_compare = float(actual_value_to_compare) if actual_value_to_compare is not None else None
               compare_value_literal = float(compare_value_literal) if compare_value_literal is not None else None
           elif value_type_hint == "boolean":
               if isinstance(actual_value_to_compare, str): actual_value_to_compare = actual_value_to_compare.lower() in ["true", "1", "yes"]
               else: actual_value_to_compare = bool(actual_value_to_compare)
               if isinstance(compare_value_literal, str): compare_value_literal = compare_value_literal.lower() in ["true", "1", "yes"]
               else: compare_value_literal = bool(compare_value_literal)
           else:
               actual_value_to_compare = str(actual_value_to_compare) if actual_value_to_compare is not None else ""
               compare_value_literal = str(compare_value_literal) if compare_value_literal is not None else ""
       except ValueError as ve:
            return {"status": "error", "evaluation": None, "message": f"Type conversion error for '{value_type_hint}': {ve}. Val: '{actual_value_to_compare}', Comp: '{compare_value_literal}'"}

       evaluation = None
       try:
           if operator == "equals": evaluation = (actual_value_to_compare == compare_value_literal)
           elif operator == "not_equals": evaluation = (actual_value_to_compare != compare_value_literal)
           elif operator == "contains":
               if isinstance(actual_value_to_compare, (str, list)): evaluation = (compare_value_literal in actual_value_to_compare)
               elif isinstance(actual_value_to_compare, dict): evaluation = (compare_value_literal in actual_value_to_compare.keys() or compare_value_literal in actual_value_to_compare.values()) # Basic dict contains check
               else: return {"status": "error", "evaluation": None, "message": f"'contains' needs str/list/dict; got {type(actual_value_to_compare)}"}
           elif operator == "not_contains":
               if isinstance(actual_value_to_compare, (str, list)): evaluation = (compare_value_literal not in actual_value_to_compare)
               elif isinstance(actual_value_to_compare, dict): evaluation = not (compare_value_literal in actual_value_to_compare.keys() or compare_value_literal in actual_value_to_compare.values())
               else: return {"status": "error", "evaluation": None, "message": f"'not_contains' needs str/list/dict; got {type(actual_value_to_compare)}"}
           elif operator == "is_true": evaluation = bool(actual_value_to_compare)
           elif operator == "is_false": evaluation = not bool(actual_value_to_compare)
           elif operator == "is_empty":
               if hasattr(actual_value_to_compare, '__len__'): evaluation = (len(actual_value_to_compare) == 0)
               else: evaluation = (actual_value_to_compare is None)
           elif operator == "is_not_empty":
               if hasattr(actual_value_to_compare, '__len__'): evaluation = (len(actual_value_to_compare) > 0)
               else: evaluation = (actual_value_to_compare is not None)
           elif operator == "greater_than":
               if isinstance(actual_value_to_compare, (int, float)) and isinstance(compare_value_literal, (int, float)): evaluation = (actual_value_to_compare > compare_value_literal)
               else: return {"status": "error", "evaluation": None, "message": "Numeric types required for 'greater_than'."}
           elif operator == "less_than":
               if isinstance(actual_value_to_compare, (int, float)) and isinstance(compare_value_literal, (int, float)): evaluation = (actual_value_to_compare < compare_value_literal)
               else: return {"status": "error", "evaluation": None, "message": "Numeric types required for 'less_than'."}
           else: return {"status": "error", "evaluation": None, "message": f"Unsupported operator: {operator}"}

           return {"status": "success", "evaluation": evaluation, "message": f"Condition '{output_var_path} {operator} {compare_value_literal}' (Actual: {actual_value_to_compare}) evaluated to {evaluation}."}
       except Exception as e:
           return {"status": "error", "evaluation": None, "message": f"Error during condition evaluation: {str(e)}"}

   def _capture_failure_context(self, plan_str:str, failed_step_def:Optional[Dict], failed_step_result:Optional[Dict], current_outputs:Dict) -> Dict:
        return {
            "plan_that_failed_this_attempt": plan_str,
            "failed_step_definition": failed_step_def if failed_step_def else "N/A",
            "failed_step_execution_result": failed_step_result if failed_step_result else "N/A",
            "step_outputs_before_failure": dict(current_outputs)
        }

   def _get_relevant_history_for_prompt(self, user_prompt:str, full_history:bool=False) -> str:
       history_to_consider = self.conversation_history[:-1] # Exclude current user prompt
       if full_history or not history_to_consider:
           relevant_turns = history_to_consider[-self.max_history_items:]
       else:
           # Simplified relevance: take last few turns if specific keyword matching isn't effective enough
           relevant_turns = history_to_consider[-4:] # e.g., last 2 user/assistant exchanges
       history_list = [f"{t['role'].capitalize()}: {str(t.get('content',''))}" for t in relevant_turns]
       history_str = "\n".join(history_list)
       return f"Relevant Conversation History:\n{history_str}\n\n" if history_str else "No relevant conversation history found.\n\n"

   def _construct_kb_query_generation_prompt(self, user_prompt:str, history_context:str, nlu_info:str) -> str:
       return ( f"{history_context}"
                f"Current User Request: '{user_prompt}'\n"
                f"NLU Analysis of Request: {nlu_info}\n\n"
                f"Your task: Based on request, history, and NLU, generate a concise search query (max 5-7 words) for a knowledge base (KB) containing general info (docs, web content, code explanations). "
                f"**Strongly consider NLU entities for a targeted query.** "
                f"If no KB query seems useful, output ONLY: NO_QUERY_NEEDED\n"
                f"Otherwise, output ONLY the query string.\nSearch Query or Marker:" )

   def _format_kb_entry_for_prompt(self, kb_hit:Dict) -> str:
       doc_preview = kb_hit.get('document','N/A')[:250]+"..."
       meta = kb_hit.get('metadata',{})
       meta_parts = [f"{k}: {str(v)[:50]}" for k,v in meta.items() if k not in ['document_content_ vezi', 'extracted_keywords', 'extracted_topics']] # Filter out long fields if needed
       meta_preview = "; ".join(meta_parts)[:150]+"..."
       kw_preview = f" (Keywords: {str(meta.get('extracted_keywords','N/A'))[:70]}...)" if meta.get("extracted_keywords") else ""
       tpc_preview = f" (Topics: {str(meta.get('extracted_topics','N/A'))[:70]}...)" if meta.get("extracted_topics") else ""
       return f"  - Content Preview: \"{doc_preview}\" (Metadata: {meta_preview}){kw_preview}{tpc_preview}"

   def _format_plan_log_entry_for_prompt(self, kb_hit:Dict) -> str:
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
       except Exception as e: print(f"Warning: Error formatting plan log for prompt: {e}"); return f"- Error processing plan log: {str(e)[:100]}"

   def _format_feedback_report_for_prompt(self, kb_hit:Dict) -> str:
        try:
            report_doc_str = kb_hit.get("document")
            if not report_doc_str: return "- Malformed feedback report (missing document)."
            report_data = json.loads(report_doc_str)
            sent_dist = report_data.get("overall_sentiment_distribution", {})
            pos_perc = sent_dist.get('positive', 0.0) * 100
            neg_perc = sent_dist.get('negative', 0.0) * 100
            insights_preview = str(report_data.get("actionable_insights", report_data.get("comment_previews", []))[:1])[:150] # Preview first insight/comment
            return (f"Feedback Insights: Overall Sentiment (Pos: {pos_perc:.1f}%, Neg: {neg_perc:.1f}%). "
                    f"Sample Insight/Comment: '{insights_preview}...'.")
        except Exception as e: print(f"Warning: Error formatting feedback report for prompt: {e}"); return f"- Error processing feedback report: {str(e)[:100]}"

   def _construct_main_planning_prompt(self, user_prompt:str, history_context:str, nlu_info:str,
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
               f"  - 'agent_name': (String) Agent from list OR 'parallel_group' OR 'conditional' (if using step_type).\n"
               f"  - 'step_type': (Optional, String) Default is agent execution. Use 'conditional' for conditional logic, or 'parallel_group' for parallel execution (then agent_name should also be 'parallel_group').\n"
               f"  - 'description': (Optional, String) Human-readable description of the step's purpose.\n"
               f"  - 'task_prompt': (String) Specific prompt for the agent. (Not used if 'parallel_group' or 'conditional').\n"
               f"  - 'dependencies': (Optional, List[String]) IDs of prior steps this step depends on.\n"
               f"  - 'output_variable_name': (Optional, String) Variable name for step's output (e.g., 'user_preference_value').\n"
               f"  - 'max_retries': (Integer, Optional, Default: 0).\n"
               f"  - 'retry_delay_seconds': (Integer, Optional, Default: 5).\n"
               f"  - 'retry_on_statuses': (List[String], Optional, Default: [\"error\"]).\n"
               f"For 'parallel_group' ('step_type':'parallel_group', 'agent_name':'parallel_group'): include 'sub_steps': [List of standard step objects]. Sub-steps MUST be input-independent. Group's 'output_variable_name' will be a dict of sub-step outputs.\n"
               f"For 'conditional' ('step_type':'conditional', 'agent_name':'conditional'): include 'condition': {{'source_step_id':'id_of_source_step', 'source_output_variable':'path.to.var_in_output', 'operator':'equals|contains|is_true|etc.', 'value':'compare_val_literal', 'value_type':'string|integer|float|boolean' (optional hint)}}, 'if_true_step_id':'id_of_next_step_if_true', 'if_false_step_id':'id_of_next_step_if_false' (optional). Ensure 'dependencies' includes 'condition.source_step_id'. Use sparingly.\n"
               f"Example Agent Step: {{'step_id': '1', 'agent_name': 'WebCrawler', 'task_prompt': 'Search for X', 'output_variable_name': 'search_X'}}\n"
               f"Example Conditional: {{'step_id': 'cond_1', 'step_type': 'conditional', 'agent_name': 'conditional', 'dependencies': ['1'], 'condition': {{'source_step_id': '1', 'source_output_variable': 'result.found_items', 'operator': 'greater_than', 'value': 0, 'value_type': 'integer'}}, 'if_true_step_id': '2', 'if_false_step_id': '3'}}\n"
               f"IMPORTANT: Output ONLY the raw JSON plan as a list of step objects. If unplannable or request is too simple for a plan, return an empty JSON list []." )

   def _construct_revision_planning_prompt(self, user_prompt:str, history_context:str, nlu_info:str,
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
               f"Original User Request: '{user_prompt}'\nNLU Analysis (from first attempt): {nlu_info}\n\n"
               f"{failure_context_section}"
               f"Available Agents (Note Complexity Ratings):\n{agent_desc}\n\n"
               f"Revision Instructions:\n1. Analyze 'DETAILED FAILURE CONTEXT' (failed plan, step, result, prior outputs).\n"
               f"2. Goal: revised JSON plan addressing failure. Make MINIMAL TARGETED changes to 'Plan that Failed This Attempt'.\n"
               f"3. Prioritize fixing/replacing failed step. Adjust subsequent steps if dependencies change.\n"
               f"4. Ensure coherence with original request. Return COMPLETE VALID JSON plan (same overall schema, including for conditional and parallel steps if used).\n"
               f"5. Consider agent complexity and if a conditional branch was involved in the failure. Could the condition be wrong, or the wrong branch taken?\n\n"
               f"IMPORTANT: Output ONLY raw JSON. If unsalvageable, return []." )

   def _capture_simple_failure_context(self, plan_str:str, last_result:Optional[Dict]) -> Dict:
       return { "plan_that_failed_this_attempt": plan_str,
                "failed_step_definition": "Capture_Error: Could not identify specific failing step definition.",
                "failed_step_execution_result": last_result or {"error":"No specific step result captured."},
                "step_outputs_before_failure": {} }

   async def _summarize_execution_for_user(self, user_prompt:str, final_exec_results:List[Dict]) -> str:
       summarizer = next((a for a in self.agents if a.name=="CreativeWriter"),None) or \
                    next((a for a in self.agents if a.name=="DeepThink"),None)
       if not summarizer:
           s_count = sum(1 for r in final_exec_results if r.get('status')=='success')
           return f"Plan execution finished. {s_count}/{len(final_exec_results)} steps processed successfully."

       summary_context = f"Original User Request: '{user_prompt}'\nPlan Execution Summary of Final Attempt:\n"
       if not final_exec_results: summary_context += "No steps were executed or the plan was empty.\n"
       else:
           for i, res in enumerate(final_exec_results):
               summary_context += (f"  Step {i+1} (Agent: {res.get('agent','N/A')}, ID: {res.get('step_id','N/A')}): Status='{res.get('status','unknown')}', "
                                   f"Output Snippet='{str(res.get('response','No response'))[:100]}...'\n")

       prompt = (f"You are an AI assistant. Based on the user's original request and a summary of the multi-step plan execution's final attempt, "
                 f"provide a concise, natural language summary of what actions were taken by the system and the overall outcome. "
                 f"Focus on what would be most useful for the user to understand what just happened. "
                 f"Do not refer to yourself as '{summarizer.name}', just act as the main AI assistant.\n\n{summary_context}\n\n"
                 f"Please provide only the summary text, suitable for conversation history.")

       res = await self.execute_agent(summarizer, prompt)
       if res.get("status")=="success" and res.get("response","").strip(): return res.get("response").strip()
       else:
           s_count = sum(1 for r in final_exec_results if r.get('status')=='success')
           return f"Plan execution attempt finished. {s_count}/{len(final_exec_results)} steps successful. Summarization failed: {res.get('response')}"

   async def _store_plan_execution_log_in_kb(self, user_prompt_orig:str, nlu_output_orig:Dict,
                                           plan_json_final_attempt:str, final_status_bool:bool,
                                           num_attempts:int, step_results_final_attempt:List[Dict],
                                           outputs_final_attempt:Dict, user_facing_summary_text:str) -> Optional[str]:
       if not self.knowledge_collection:
           print("MasterPlanner: Knowledge Base unavailable, skipping storage of plan execution summary.")
           return None

       final_plan_status_str = "success" if final_status_bool else "failure"
       summary_list_for_log = [{"step_id":s.get("step_id","N/A"), "agent_name":s.get("agent","N/A"),
                                "status":s.get("status","unknown"), "response_preview":str(s.get("response",""))[:150]+"..."}
                               for s in step_results_final_attempt]

       nlu_analysis_data = {}
       if isinstance(nlu_output_orig, dict):
           nlu_analysis_data = { "intent":nlu_output_orig.get("intent"),
                                 "intent_scores":nlu_output_orig.get("intent_scores"),
                                 "entities":nlu_output_orig.get("entities",[]) }

       summary_dict = {
           "version":"1.1_conditional", "original_user_request":user_prompt_orig,
           "nlu_analysis_on_request": nlu_analysis_data,
           "plan_json_executed_final_attempt":plan_json_final_attempt,
           "execution_summary":{ "overall_status":final_plan_status_str, "total_attempts":num_attempts,
                                 "final_attempt_step_results":summary_list_for_log,
                                 "outputs_from_successful_steps_final_attempt": {k: (str(v)[:200]+"..." if len(str(v)) > 200 else v) for k,v in outputs_final_attempt.items()} },
           "user_facing_plan_outcome_summary":user_facing_summary_text,
           "log_timestamp_iso":datetime.datetime.now().isoformat()
       }
       content_str = json.dumps(summary_dict, indent=2)

       kb_meta = { "source":"plan_execution_log", "overall_status":final_plan_status_str,
                   "user_request_preview":user_prompt_orig[:150],
                   "primary_intent":nlu_analysis_data.get("intent","N/A"),
                   "log_timestamp_iso":summary_dict["log_timestamp_iso"] }

       if nlu_analysis_data.get("entities"):
           for i, ent in enumerate(nlu_analysis_data["entities"][:3]):
               kb_meta[f"entity_{i+1}_type"]=ent.get("type","UNK")
               kb_meta[f"entity_{i+1}_text"]=str(ent.get("text",""))[:50]

       kb_store_coro = self.store_knowledge(content_str, kb_meta)

       stored_kb_id = None
       async def _publish_after_plan_log_store():
           nonlocal stored_kb_id
           kb_res = await kb_store_coro
           if kb_res.get("status")=="success":
               stored_kb_id = kb_res.get("id")
               await self.publish_message("kb.plan_execution_log.added", "MasterPlanner",
                   payload={"kb_id":stored_kb_id, "original_request_preview":user_prompt_orig[:150],
                            "overall_status":final_plan_status_str, "primary_intent":nlu_analysis_data.get("intent","N/A")})

       await _publish_after_plan_log_store()
       print(f"MasterPlanner: Plan log storage and publish task processing finished. Stored KB ID: {stored_kb_id}")
       return stored_kb_id

# ... other methods ...

# Ensure the instantiations are at the very end after all class definitions
# (Assuming DocumentUniverse and WebIntelligence are defined above this or correctly imported)
# orchestrator = TerminusOrchestrator() # This line should be removed if class definition is above
# doc_processor = DocumentUniverse()
# web_intel = WebIntelligence()
# auto_dev is imported

class DocumentUniverse:
   def process_file(self, uploaded_file: Any) -> str: # uploaded_file is a Streamlit UploadedFile like object
        file_name = getattr(uploaded_file, 'name', 'unknown_file')
        file_ext = Path(file_name).suffix.lower().strip('.')

        try:
            if file_ext == 'txt':
                return uploaded_file.read().decode('utf-8', errors='replace')
            elif file_ext == 'json':
                return json.dumps(json.load(uploaded_file))
            elif file_ext == 'csv':
                return uploaded_file.read().decode('utf-8', errors='replace')
            else:
                return f"Content from '{file_name}' (type: {file_ext}) - processing placeholder. Full parsing requires additional libraries for this type."
        except Exception as e:
            return f"Error processing file {file_name}: {str(e)}"

class WebIntelligence:
    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        print(f"Simulating web search for: {query} (returning {num_results} placeholder results)")
        results = []
        for i in range(num_results):
            results.append({
                "title": f"Placeholder Search Result {i+1} for '{query}'",
                "url": f"http://example.com/search?q={query.replace(' ', '+')}&page={i+1}",
                "snippet": f"This is a placeholder snippet for search result {i+1} related to '{query}'. More details would be here."
            })
        return results

    def scrape_page(self, url: str) -> Dict:
        print(f"Simulating scraping page: {url}")
        if "example.com" in url: # Simplified mock
            return { "status": "success", "content": f"Placeholder scraped content for {url}.", "url": url, "full_content_length": 200 }
        else:
            return { "status": "error", "message": f"Mock scrape failed for {url}.", "url": url }

orchestrator = TerminusOrchestrator()
doc_processor = DocumentUniverse()
web_intel = WebIntelligence()
# auto_dev instance is created in its own file (src/tools/auto_dev.py) and imported.

[end of src/agents/master_orchestrator.py]
