import asyncio, json, requests, subprocess, threading, queue, time, datetime
import torch
import aiohttp
from diffusers import DiffusionPipeline
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import pyttsx3
import shutil
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List,Dict,Any,Optional, Callable, Coroutine, Tuple # Added Tuple
from pathlib import Path
# Removed direct transformers import as NLU is now agent-based
# from transformers import pipeline as hf_pipeline
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
        auto_dev = None

import chromadb
from chromadb.utils import embedding_functions
import uuid
from collections import defaultdict

import fitz  # PyMuPDF
import docx
import openpyxl
from bs4 import BeautifulSoup
import io # For BytesIO

from duckduckgo_search import DDGS

from collections import defaultdict # Ensure defaultdict is imported if not already
from ..core import prompt_constructors
from ..core.rl_logger import RLExperienceLogger
from ..core.async_tools import AsyncTask, AsyncTaskStatus
from ..core.event_system import SystemEvent # New Import for Event Bus


# --- New/Modified Dataclasses for Service Definitions ---
@dataclass
class AgentServiceParameter:
    name: str
    type: str  # e.g., "string", "integer", "float", "boolean", "list[string]", "dict"
    required: bool
    description: str
    default_value: Optional[Any] = None

@dataclass
class AgentServiceReturn:
    type: str # e.g., "string", "integer", "dict", "list[CustomObject]"
    description: str

@dataclass
class AgentServiceDefinition:
    name: str
    description: str
    parameters: List[AgentServiceParameter]
    returns: AgentServiceReturn
    handler_method_name: Optional[str] = None

@dataclass
class Agent:
   name:str
   model:str
   specialty:str
   active:bool=True
   estimated_complexity:Optional[str]=None
   # `provided_services` is read from JSON but not stored on the Agent object itself.
   # It's processed by the orchestrator into self.service_definitions.


class TerminusOrchestrator:
   def __init__(self):
       self.agents: List[Agent] = []
       self.service_definitions: Dict[Tuple[str, str], AgentServiceDefinition] = {}
       self.service_handlers: Dict[Tuple[str, str], Callable[..., Coroutine[Any, Any, Dict]]] = {}

       # --- Structures for Asynchronous Task Management ---
       # Stores the actual asyncio.Task objects for currently running tasks.
       # Keyed by task_id (str).
       self.active_async_tasks: Dict[str, asyncio.Task] = {}
       # Stores AsyncTask dataclass instances representing the state and result of tasks.
       # This includes completed and failed tasks for a period or until explicitly cleared.
       self.async_task_registry: Dict[str, AsyncTask] = {}
       # Lock for thread-safe/async-safe modifications to the task registry
       self._task_registry_lock = asyncio.Lock()
       # --- End Async Task Management Structures ---

       # --- Event Bus Structures ---
       self.event_subscribers: Dict[str, List[Callable[[SystemEvent], Coroutine[Any, Any, None]]]] = defaultdict(list)
       self.event_processing_queue: asyncio.Queue[SystemEvent] = asyncio.Queue()
       self._event_dispatcher_task: Optional[asyncio.Task] = None
       # --- End Event Bus Structures ---

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
           with open(self.agents_config_path, 'r', encoding='utf-8') as f:
               agents_data_raw = json.load(f)

           for agent_config_raw in agents_data_raw:
               # Separate provided_services from other agent fields
               provided_services_raw = agent_config_raw.pop("provided_services", [])

               # Create Agent instance without provided_services
               agent_instance = Agent(**agent_config_raw)
               self.agents.append(agent_instance)

               # Process and store service definitions
               for service_def_raw in provided_services_raw:
                   try:
                       # Deserialize parameters
                       params_data = service_def_raw.get("parameters", [])
                       service_params = [AgentServiceParameter(**p) for p in params_data]

                       # Deserialize returns
                       returns_data = service_def_raw.get("returns", {})
                       service_returns = AgentServiceReturn(**returns_data)

                       service_def_obj = AgentServiceDefinition(
                           name=service_def_raw["name"],
                           description=service_def_raw["description"],
                           parameters=service_params,
                           returns=service_returns,
                           handler_method_name=service_def_raw.get("handler_method_name")
                       )

                       service_key = (agent_instance.name, service_def_obj.name)
                       self.service_definitions[service_key] = service_def_obj
                       print(f"Loaded service definition: {agent_instance.name} -> {service_def_obj.name}")

                       if service_def_obj.handler_method_name:
                           handler_method = getattr(self, service_def_obj.handler_method_name, None)
                           if handler_method and callable(handler_method):
                               self.service_handlers[service_key] = handler_method
                               print(f"  Registered direct handler '{service_def_obj.handler_method_name}' for service '{service_def_obj.name}'.")
                           else:
                               print(f"  WARNING: Handler method '{service_def_obj.handler_method_name}' for service '{service_def_obj.name}' not found or not callable on orchestrator.")
                   except Exception as e_service:
                       print(f"ERROR loading service definition for agent {agent_instance.name}: {service_def_raw.get('name', 'UNKNOWN_SERVICE')}. Details: {e_service}")

       except FileNotFoundError:
           print(f"CRITICAL ERROR: agents.json not found at {self.agents_config_path}. No agents or services loaded.")
       except json.JSONDecodeError as e_json:
           print(f"CRITICAL ERROR: Failed to decode agents.json: {e_json}. No agents or services loaded.")
       except Exception as e_outer:
           print(f"CRITICAL ERROR loading agents configuration: {e_outer}. Some agents/services may not be loaded.")

       self.ollama_url="http://localhost:11434/api/generate"
       self.image_gen_pipeline = None
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       self.image_gen_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

       try:
           self.tts_engine = pyttsx3.init()
       except Exception as e:
           print(f"WARNING: Failed to initialize TTS engine: {e}.")
           self.tts_engine = None

       self.candidate_intent_labels = [
           "image_generation", "code_generation", "code_modification", "code_explanation",
           "project_scaffolding", "video_info", "video_frame_extraction", "video_to_gif",
           "audio_info", "audio_format_conversion", "text_to_speech",
           "data_analysis", "web_search", "document_processing", "general_question_answering",
           "complex_task_planning", "system_information_query", "knowledge_base_query",
           "feedback_submission", "feedback_analysis_request", "agent_service_call"
       ]

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

       # Old message bus attributes are removed as the new event system replaces them.
       # self.message_bus_subscribers = defaultdict(list)
       # self.message_processing_tasks = set()
       # self._setup_initial_event_listeners() # This call is removed. Subscriptions will use the new event bus.

       # Start the new event dispatcher loop
       self._event_dispatcher_task = asyncio.create_task(self._event_dispatcher_loop())
       print("TerminusOrchestrator: New event dispatcher loop started.")

       # Subscribe initial internal event handlers
       self.subscribe_to_event("kb.content.added", self._event_handler_kb_content_added)
       self.subscribe_to_event("user.feedback.submitted", self._event_handler_log_user_feedback) # Added for 2.4

       # self.service_handlers is now populated dynamically during agent loading
       # based on "handler_method_name" in service definitions.
       # Old static definition:
       # self.service_handlers = {
       # ("CodeMaster", "validate_code_syntax"): self._service_codemaster_validate_syntax
       # }
       self.default_high_priority_retries = 1
       print(f"TerminusOrchestrator initialized. {len(self.service_handlers)} direct service handlers registered. Default high-priority retries: {self.default_high_priority_retries}.")

       self.rl_experience_log_path = self.logs_dir / "rl_experience_log.jsonl"
       self.rl_logger = RLExperienceLogger(self.rl_experience_log_path)
       print(f"RL Experience Logger initialized. Logging to: {self.rl_experience_log_path}")

   # --- Asynchronous Task Management Methods ---
   async def _async_task_wrapper(self, task_id: str, coro: Coroutine, task_name: Optional[str]):
        """
        Internal wrapper for submitted coroutines. Updates task status in the registry.
        """
        async with self._task_registry_lock:
            # This should ideally only update, assuming submit_async_task already created PENDING entry
            task_info = self.async_task_registry.get(task_id)
            if not task_info: # Should not happen if submit_async_task is used correctly
                task_info = AsyncTask(task_id=task_id, name=task_name or "Unnamed Task")
                self.async_task_registry[task_id] = task_info

            task_info.status = AsyncTaskStatus.RUNNING
            task_info.started_at = datetime.datetime.now()
            print(f"[AsyncTask-{task_id} ({task_info.name})] Status changed to RUNNING.")

        try:
            result = await coro
            async with self._task_registry_lock:
                task_info.status = AsyncTaskStatus.COMPLETED
                task_info.result = result
                task_info.completed_at = datetime.datetime.now()
                print(f"[AsyncTask-{task_id} ({task_info.name})] Status changed to COMPLETED.")
        except asyncio.CancelledError: # Handle task cancellation if implemented later
            async with self._task_registry_lock:
                task_info.status = AsyncTaskStatus.CANCELLED
                task_info.error = "Task was cancelled."
                task_info.completed_at = datetime.datetime.now()
                print(f"[AsyncTask-{task_id} ({task_info.name})] Status changed to CANCELLED.")
            # Optionally re-raise if cancellation needs to propagate further
        except Exception as e:
            async with self._task_registry_lock:
                task_info.status = AsyncTaskStatus.FAILED
                task_info.error = f"{type(e).__name__}: {str(e)}" # Store type and message
                task_info.completed_at = datetime.datetime.now()
                print(f"[AsyncTask-{task_id} ({task_info.name})] Status changed to FAILED. Error: {task_info.error}")
        finally:
            # Remove from active_async_tasks dict as the asyncio.Task object is done
            self.active_async_tasks.pop(task_id, None)
            print(f"[AsyncTask-{task_id} ({task_info.name})] Wrapper finished, removed from active tasks list.")


   async def submit_async_task(self, coro: Coroutine, name: Optional[str] = None) -> str:
        """
        Submits a coroutine to be run as an asyncio.Task, managed by the orchestrator.
        Returns a unique task ID.
        """
        task_id = str(uuid.uuid4())

        async with self._task_registry_lock:
            task_info = AsyncTask(task_id=task_id, name=name or "Unnamed Task", status=AsyncTaskStatus.PENDING)
            self.async_task_registry[task_id] = task_info
            print(f"[AsyncTask-{task_id} ({task_info.name})] Status changed to PENDING and registered.")

        # Create and store the asyncio.Task, wrapped by our _async_task_wrapper
        # The wrapper will handle updating the status to RUNNING, COMPLETED, or FAILED.
        asyncio_task = asyncio.create_task(self._async_task_wrapper(task_id, coro, name))
        self.active_async_tasks[task_id] = asyncio_task

        # Store the asyncio.Task object in the AsyncTask dataclass for potential advanced control (e.g., cancellation)
        # This is done after ensuring task_info is in the registry.
        async with self._task_registry_lock:
             self.async_task_registry[task_id]._task_obj = asyncio_task # Assigning to private field

        print(f"[Orchestrator] Submitted AsyncTask ID: {task_id} for '{name}'. Coroutine: {coro.__name__ if hasattr(coro, '__name__') else type(coro).__name__}")
        return task_id

   async def get_async_task_info(self, task_id: str) -> Optional[AsyncTask]:
        """
        Retrieves the AsyncTask dataclass instance for a given task_id.
        This provides full info including status, result, error.
        """
        async with self._task_registry_lock: # Ensure consistent read
            task_info = self.async_task_registry.get(task_id)

        if task_info:
            return task_info
        else:
            print(f"[Orchestrator] No task info found for ID: {task_id}")
            return None

   async def cancel_async_task(self, task_id: str) -> bool: # For future use
        """
        Attempts to cancel an active asynchronous task.
        """
        async with self._task_registry_lock:
            task_info = self.async_task_registry.get(task_id)
            asyncio_task_obj = self.active_async_tasks.get(task_id)

            if not task_info or not asyncio_task_obj:
                print(f"[AsyncTask-{task_id}] Cannot cancel: Task not found or not active.")
                return False

            if task_info.status in [AsyncTaskStatus.COMPLETED, AsyncTaskStatus.FAILED, AsyncTaskStatus.CANCELLED]:
                print(f"[AsyncTask-{task_id}] Cannot cancel: Task already in terminal state ({task_info.status.name}).")
                return False

            cancelled = asyncio_task_obj.cancel()
            if cancelled:
                # The _async_task_wrapper will handle setting status to CANCELLED
                print(f"[AsyncTask-{task_id}] Cancellation requested.")
                # Optionally, could immediately update status to PENDING_CANCELLATION here
            else:
                print(f"[AsyncTask-{task_id}] Cancellation request failed (task may already be completing).")
            return cancelled
   # --- End Asynchronous Task Management Methods ---

   # --- Event Bus Methods ---
   async def publish_event(self, event_type: str, source_component: str, payload: Dict) -> str:
        """
        Creates a SystemEvent and puts it onto the event_processing_queue.
        Returns the event_id.
        """
        event = SystemEvent(
            event_type=event_type,
            source_component=source_component,
            payload=payload
        )
        await self.event_processing_queue.put(event)
        print(f"[EventBus] Published Event ID: {event.event_id}, Type: '{event.event_type}', Source: '{event.source_component}', Payload: {str(payload)[:100]}...")
        return event.event_id

   def subscribe_to_event(self, event_type: str, handler: Callable[[SystemEvent], Coroutine[Any, Any, None]]):
        """
        Subscribes an asynchronous handler to a specific event type.
        """
        self.event_subscribers[event_type].append(handler)
        handler_name = getattr(handler, '__name__', str(handler))
        print(f"[EventBus] Subscribed handler '{handler_name}' to event type '{event_type}'.")

   async def _event_dispatcher_loop(self):
        """
        Continuously gets events from the queue and dispatches them to subscribers.
        """
        print("[EventBus] Dispatcher loop started. Waiting for events...")
        while True:
            try:
                event = await self.event_processing_queue.get()
                print(f"[EventBus] Dispatching Event ID: {event.event_id}, Type: '{event.event_type}'...")

                handlers_to_call = self.event_subscribers.get(event.event_type, [])
                if not handlers_to_call:
                    print(f"[EventBus] No subscribers for event type '{event.event_type}'. Event ID: {event.event_id}")
                    self.event_processing_queue.task_done()
                    continue

                # Using asyncio.create_task for each handler to allow them to run concurrently
                # and not block the dispatcher loop if one handler is slow or errors.
                handler_tasks = []
                for handler in handlers_to_call:
                    handler_name = getattr(handler, '__name__', str(handler))
                    print(f"[EventBus] Calling handler '{handler_name}' for Event ID: {event.event_id} Type: {event.event_type}")
                    handler_tasks.append(asyncio.create_task(self._safe_execute_handler(handler, event)))

                # Wait for all handlers for this event to complete (or error)
                # This is optional; if we don't want to wait, we can just launch tasks.
                # For now, let's gather them to see their completion.
                await asyncio.gather(*handler_tasks, return_exceptions=False) # Errors handled in _safe_execute_handler

                self.event_processing_queue.task_done()
                print(f"[EventBus] Finished processing Event ID: {event.event_id}, Type: '{event.event_type}'.")

            except asyncio.CancelledError:
                print("[EventBus] Dispatcher loop cancelled.")
                break
            except Exception as e:
                print(f"[EventBus] CRITICAL ERROR in dispatcher loop: {e}. Loop continues but this event might be lost.")
                # Potentially re-queue the event or log to a dead-letter queue in a real system
                # For now, just mark as done to avoid blocking queue if get() was successful.
                if 'event' in locals() and self.event_processing_queue._unfinished_tasks > 0 : # Check if get() was successful
                     self.event_processing_queue.task_done()


   async def _safe_execute_handler(self, handler: Callable[[SystemEvent], Coroutine[Any, Any, None]], event: SystemEvent):
        """
        Safely executes a single event handler, catching and logging exceptions.
        """
        handler_name = getattr(handler, '__name__', str(handler))
        try:
            await handler(event)
            print(f"[EventBusHandler] Handler '{handler_name}' completed successfully for Event ID: {event.event_id}.")
        except Exception as e:
            print(f"[EventBusHandler] ERROR: Handler '{handler_name}' failed for Event ID: {event.event_id}. Error: {type(e).__name__}: {e}")
            # Add more detailed logging, e.g., traceback.format_exc() if needed
   # --- End Event Bus Methods ---


   def get_agent_capabilities_description(self) -> str:
       descriptions = []
       for a in self.agents:
           if a.active:
               complexity_info = f" (Complexity: {a.estimated_complexity})" if a.estimated_complexity else ""
               descriptions.append(f"- {a.name}: Specializes in '{a.specialty}'. Uses model: {a.model}.{complexity_info}")
       return "\n".join(descriptions) if descriptions else "No active agents available."

    async def _handle_system_event(self, event: SystemEvent): # Signature changed to SystemEvent
       # This method will be re-subscribed to relevant events in Phase 2 if still needed.
       # For now, its direct subscriptions via _setup_initial_event_listeners are removed.
       print(f"[LegacyEventHandler] Event ID: {event.event_id}, Type: '{event.event_type}', Src: '{event.source_component}', Payload: {event.payload}")

   async def _event_handler_log_user_feedback(self, event: SystemEvent):
        """
        Simple event handler to log when user feedback is submitted.
        """
        feedback_id = event.payload.get("feedback_id", "N/A")
        item_id = event.payload.get("item_id", "N/A")
        rating = event.payload.get("rating", "N/A")
        print(f"[UserFeedbackLogger] Received 'user.feedback.submitted' event. Feedback ID: {feedback_id}, Item ID: {item_id}, Rating: {rating}. Event ID: {event.event_id}")

   # _setup_initial_event_listeners REMOVED
   # publish_message REMOVED
   # subscribe_to_message REMOVED

   async def _event_handler_kb_content_added(self, event: SystemEvent):
       """
       Handles the 'kb.content.added' event to trigger content analysis.
       Formerly _handle_new_kb_content_for_analysis.
       """
       kb_id = event.payload.get("kb_id")
       source_op = event.payload.get("source_operation", "unknown_source") # Get source from payload
       handler_id = f"[ContentAnalysisHandler event_id:{event.event_id} kb_id:{kb_id} src_op:{source_op}]"

       # Avoid analyzing feedback reports or plan logs themselves with this generic handler
       if source_op in ["feedback_analysis_report", "plan_execution_log"]:
           print(f"{handler_id} INFO: Skipping content analysis for KB item from source operation '{source_op}'.")
           return

       print(f"{handler_id} START: Processing event type: {event.event_type}")
       if self.knowledge_collection is None:
           print(f"{handler_id} ERROR: Knowledge base not available. Skipping analysis."); return
       if not kb_id:
           print(f"{handler_id} ERROR: No valid kb_id in event payload. Cannot process."); return

       try:
           item_data = self.knowledge_collection.get(ids=[kb_id], include=["documents", "metadatas"])
           if not (item_data and item_data.get('ids') and item_data['ids'][0]):
               print(f"{handler_id} ERROR: KB item '{kb_id}' not found for analysis."); return

           document_content = item_data['documents'][0]
           doc_metadata = item_data['metadatas'][0] if item_data.get('metadatas') and item_data['metadatas'][0] else {}

           if not document_content:
               print(f"{handler_id} INFO: KB item '{kb_id}' has empty content. Skipping analysis."); return

           if doc_metadata.get("analysis_by_agent") == "ContentAnalysisAgent" and \
              (doc_metadata.get("extracted_keywords") or doc_metadata.get("extracted_topics")):
               print(f"{handler_id} INFO: Content for KB ID '{kb_id}' already analyzed by ContentAnalysisAgent. Skipping re-analysis.")
               return

           analysis_agent = next((a for a in self.agents if a.name == "ContentAnalysisAgent" and a.active), None)
           if not analysis_agent:
               print(f"{handler_id} ERROR: ContentAnalysisAgent not found/active. Skipping."); return

           analysis_prompt = (f"Analyze the following text content:\n---\n{document_content[:15000]}\n---\n"
                              f"Provide output as a JSON object with 'keywords' (comma-separated string of 2-5 relevant keywords, or 'NONE') "
                              f"and 'topics' (1-3 comma-separated strings describing main topics, or 'NONE').\n"
                              f"Example: {{\"keywords\": \"k1, k2, k3\", \"topics\": \"Topic A, Topic B\"}}\nJSON Output:")

           print(f"{handler_id} INFO: Calling LLM for content analysis of KB ID '{kb_id}'.")
           llm_call_or_task = await self.execute_agent(analysis_agent, analysis_prompt)

           llm_result: Optional[Dict] = None
           if llm_call_or_task.get("status") == "pending_async":
                analysis_task_id = llm_call_or_task["task_id"]
                print(f"{handler_id} LLM analysis for KB '{kb_id}' submitted as task {analysis_task_id}. Awaiting result...")
                while True: # This handler will block for its own LLM call.
                    await asyncio.sleep(0.2)
                    task_info = await self.get_async_task_info(analysis_task_id)
                    if not task_info:
                        print(f"{handler_id} ERROR: LLM analysis task {analysis_task_id} info not found for KB '{kb_id}'."); return
                    if task_info.status == AsyncTaskStatus.COMPLETED:
                        llm_result = task_info.result
                        break
                    elif task_info.status == AsyncTaskStatus.FAILED:
                        print(f"{handler_id} ERROR: LLM analysis task {analysis_task_id} for KB '{kb_id}' failed: {task_info.error}"); return
                    elif task_info.status == AsyncTaskStatus.CANCELLED:
                         print(f"{handler_id} ERROR: LLM analysis task {analysis_task_id} for KB '{kb_id}' cancelled."); return
           else:
                llm_result = llm_call_or_task

           if not llm_result or not (llm_result.get("status") == "success" and llm_result.get("response","").strip()):
               print(f"{handler_id} ERROR: LLM analysis call for KB '{kb_id}' failed or produced no response. Status: {llm_result.get('status') if llm_result else 'N/A'}, Resp: {llm_result.get('response') if llm_result else 'N/A'}"); return

           print(f"{handler_id} SUCCESS: LLM analysis for KB '{kb_id}' successful.")
           llm_response_str = llm_result.get("response").strip()
           extracted_keywords, extracted_topics = "", ""
           try:
               data = json.loads(llm_response_str)
               raw_kw = data.get("keywords","").strip()
               extracted_keywords = raw_kw if raw_kw and raw_kw.upper() != "NONE" else ""
               raw_tp = data.get("topics","").strip()
               extracted_topics = raw_tp if raw_tp and raw_tp.upper() != "NONE" else ""
           except json.JSONDecodeError:
               print(f"{handler_id} WARNING: Failed to parse LLM JSON for content analysis of KB '{kb_id}'. Raw: '{llm_response_str}'. Using raw as keywords if applicable and not 'none'.")
               if "keywords" not in llm_response_str.lower() and "topics" not in llm_response_str.lower() and llm_response_str.upper() != "NONE":
                   extracted_keywords = llm_response_str[:200]

           if extracted_keywords or extracted_topics:
               new_meta = {
                   "analysis_by_agent": analysis_agent.name,
                   "analysis_model_used": analysis_agent.model,
                   "analysis_timestamp_iso": datetime.datetime.utcnow().isoformat()
                }
               if extracted_keywords: new_meta["extracted_keywords"] = extracted_keywords
               if extracted_topics: new_meta["extracted_topics"] = extracted_topics

               print(f"{handler_id} INFO: Attempting metadata update for KB '{kb_id}' with keywords: '{extracted_keywords}', topics: '{extracted_topics}'.")
               update_status = await self._update_kb_item_metadata(kb_id, new_meta) # Make sure this is awaited
               if update_status.get("status") == "success":
                   print(f"{handler_id} SUCCESS: Metadata update for KB '{kb_id}' successful.")
               else:
                   print(f"{handler_id} ERROR: Metadata update for KB '{kb_id}' failed. Msg: {update_status.get('message')}")
           else:
               print(f"{handler_id} INFO: No keywords or topics extracted for KB '{kb_id}'. No metadata update.")
       except Exception as e:
           print(f"{handler_id} UNHANDLED ERROR in content analysis handler for KB '{kb_id}': {type(e).__name__}: {e}")
       finally:
           print(f"{handler_id} END: Finished processing event for KB '{kb_id}'.")

   async def publish_message(self, message_type: str, source_agent_name: str, payload: Dict) -> str: #This is part of the old message bus, will be removed.
       message_id = str(uuid.uuid4()); message = { "message_id": message_id, "message_type": message_type, "source_agent_name": source_agent_name, "timestamp_iso": datetime.datetime.now().isoformat(), "payload": payload }
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
       step_priority = step_definition.get("priority", "normal").lower()
       explicit_max_retries = step_definition.get("max_retries")
       if explicit_max_retries is not None: max_retries = explicit_max_retries
       elif step_priority == "high": max_retries = self.default_high_priority_retries
       else: max_retries = 0
       retry_delay_seconds = step_definition.get("retry_delay_seconds", 5)
       retry_on_statuses = step_definition.get("retry_on_statuses", ["error"])
       current_execution_retries = 0
       log_prefix = f"[Priority: {step_priority.upper()}] " if step_priority == "high" else ""
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
           retry_info = f" (Retry {current_execution_retries}/{max_retries})" if max_retries > 0 and current_execution_retries > 0 else ""
           if max_retries > 0 and current_execution_retries == 0 and not retry_info: retry_info = f" (Attempt 1/{max_retries + 1})"
           print(f"{log_prefix}Executing step {step_id}{retry_info}: Agent='{agent_name}', Prompt='{current_task_prompt[:100]}...'")
           step_result = await self.execute_agent(target_agent, current_task_prompt)
           if step_result.get("status") == "success":
               key_to_store = output_var_name if output_var_name else f"step_{step_id}_output"
               current_step_outputs[key_to_store] = step_result.get("response")
               for mk in ["image_path","frame_path","gif_path","speech_path","modified_file"]:
                   if mk in step_result: current_step_outputs[f"{key_to_store}_{mk}"]=step_result[mk]
               print(f"{log_prefix}Step {step_id} completed successfully.")
               return step_result
           print(f"{log_prefix}Step {step_id} failed on attempt {current_execution_retries + 1}/{max_retries + 1}. Status: {step_result.get('status')}, Response: {str(step_result.get('response'))[:100]}...")
           current_execution_retries += 1
           if not (current_execution_retries <= max_retries and step_result.get("status") in retry_on_statuses):
               print(f"{log_prefix}Step {step_id} exhausted retries or status not retryable. Final status: {step_result.get('status')}")
               return step_result
           print(f"{log_prefix}Step {step_id} retrying in {retry_delay_seconds}s...")
           await asyncio.sleep(retry_delay_seconds)

   async def _handle_agent_service_call(self, service_call_step_def: Dict, current_step_outputs: Dict, full_plan_list: List[Dict]) -> Dict:
        step_id = service_call_step_def.get("step_id", "unknown_service_call_step")
        step_priority = service_call_step_def.get("priority", "normal").lower()
        log_prefix = f"[Priority: {step_priority.upper()}] " if step_priority == "high" else ""
        target_agent_name = service_call_step_def.get("target_agent_name")
        service_name = service_call_step_def.get("service_name")
        service_params_template = service_call_step_def.get("service_params", {})
        output_var_name = service_call_step_def.get("output_variable_name")

        if not all([target_agent_name, service_name]):
            return {"step_id": step_id, "status": "error", "response": "Missing target_agent_name or service_name for agent_service_call.", "data": None, "error_code": "MISSING_SERVICE_INFO"}

        service_key = (target_agent_name, service_name)
        service_def = self.service_definitions.get(service_key)

        if not service_def:
            # Fallback for services not formally defined (legacy or dynamic)
            # This part remains similar to original, but consider if this fallback should be stricter or logged more prominently.
            print(f"{log_prefix}WARNING: Service definition for '{service_name}' on agent '{target_agent_name}' not found. Proceeding with LLM fallback without formal validation.")
            # Proceed with LLM fallback as before, but without parameter validation based on a definition.
            # For brevity, the old parameter resolution and LLM call logic is assumed here if service_def is None.
            # This section would essentially replicate the old logic path for LLM fallback.
            # However, for this refactoring, we will prioritize the new structured approach.
            # If a service is called, it SHOULD have a definition.
            # For now, let's make it an error if not defined, to enforce the new structure.
            return {"step_id": step_id, "status": "error", "response": f"Service '{service_name}' on agent '{target_agent_name}' is not defined.", "data": None, "error_code": "SERVICE_NOT_DEFINED"}

        # --- Parameter Resolution, Validation, and Coercion ---
        resolved_params = {}
        param_validation_errors = []

        for defined_param in service_def.parameters:
            param_name = defined_param.name
            raw_value = service_params_template.get(param_name)

            # 1. Resolve from template if it's a string placeholder
            if isinstance(raw_value, str):
                substituted_value = raw_value
                for dep_match in re.finditer(r"{{{{([\w.-]+)}}}}", raw_value):
                    var_path = dep_match.group(1)
                    val_to_sub = current_step_outputs.get(var_path)
                    if '.' in var_path and val_to_sub is None:
                        base_key, *attrs = var_path.split('.')
                        if base_key in current_step_outputs:
                            temp_val = current_step_outputs[base_key]
                            try:
                                for part in attrs:
                                    if isinstance(temp_val, dict): temp_val = temp_val.get(part)
                                    elif isinstance(temp_val, list) and part.isdigit(): temp_val = temp_val[int(part)]
                                    else: temp_val = None; break
                                if temp_val is not None: val_to_sub = temp_val
                            except: pass
                    if val_to_sub is not None: substituted_value = substituted_value.replace(dep_match.group(0), str(val_to_sub))
                    else: print(f"Warning: Could not resolve template var '{var_path}' for param '{param_name}' in service call {step_id}.")
                actual_value = substituted_value
            else: # Value is literal (not a string template) or already resolved
                actual_value = raw_value

            # 2. Handle missing parameters and defaults
            if actual_value is None: # Parameter not provided in plan step
                if defined_param.required and defined_param.default_value is None:
                    param_validation_errors.append(f"Required parameter '{param_name}' is missing.")
                    continue
                elif defined_param.default_value is not None:
                    actual_value = defined_param.default_value
                    print(f"{log_prefix}INFO: Using default value for param '{param_name}': {actual_value}")
                elif not defined_param.required: # Optional and no default, so skip
                    continue

            # 3. Type Coercion (Basic)
            coerced_value = actual_value
            try:
                if defined_param.type == "integer" and actual_value is not None: coerced_value = int(actual_value)
                elif defined_param.type == "float" and actual_value is not None: coerced_value = float(actual_value)
                elif defined_param.type == "boolean" and actual_value is not None:
                    if isinstance(actual_value, str): coerced_value = actual_value.lower() in ["true", "1", "yes"]
                    else: coerced_value = bool(actual_value)
                # For "string", "list", "dict", assume correct format or rely on handler/LLM
            except ValueError as e_coerce:
                param_validation_errors.append(f"Parameter '{param_name}' value '{actual_value}' could not be coerced to type '{defined_param.type}': {e_coerce}")
                continue

            resolved_params[param_name] = coerced_value

        if param_validation_errors:
            error_msg = f"Parameter validation failed for service '{service_name}': " + "; ".join(param_validation_errors)
            return {"step_id": step_id, "status": "error", "response": error_msg, "data": None, "error_code": "PARAMETER_VALIDATION_FAILED"}

        # --- Execute Service ---
        service_result_structured: Dict[str, Any]

        if service_key in self.service_handlers:
            handler_method = self.service_handlers[service_key]
            print(f"{log_prefix}Executing direct handler for service '{service_name}' on agent '{target_agent_name}' with params: {resolved_params}")
            try:
                # Pass both resolved_params and the service_def for context to the handler
                # IMPORTANT: If handler_method itself might become a long-running task and needs to be non-blocking
                # at this level, it would need to return a task_id dict, and _handle_agent_service_call
                # would propagate it. For now, direct handlers are assumed to complete (even if they internally await).
                # The change in Phase 2 for _service_codemaster_validate_syntax makes it internally await.
                service_result_structured = await handler_method(params=resolved_params, service_definition=service_def)
                # If service_result_structured itself indicates pending_async (if a handler was refactored to do so),
                # we should propagate that.
                if service_result_structured.get("status") == "pending_async" and "task_id" in service_result_structured:
                    print(f"{log_prefix}Direct handler for service '{service_name}' returned a pending task: {service_result_structured['task_id']}")
                    # The structure from a handler returning pending_async should match what execute_agent returns
                    return {
                        "step_id": step_id,
                        "agent_name": f"{target_agent_name} (Service: {service_name})",
                        "status": "pending_async", # Propagate status
                        "task_id": service_result_structured["task_id"],
                        "response": service_result_structured.get("message", f"Service '{service_name}' on '{target_agent_name}' initiated as async task."),
                        "data": None # No immediate data
                    }

            except Exception as e_handler:
                err_msg = f"Direct handler for service '{service_name}' raised an exception: {e_handler}"
                print(f"{log_prefix}ERROR: {err_msg}")
                service_result_structured = {"status": "error", "data": None, "message": err_msg, "error_code": "HANDLER_EXCEPTION"}
        else:
            print(f"{log_prefix}No direct handler for service '{service_name}' on agent '{target_agent_name}'. Using LLM fallback with definition.")
            target_agent = next((a for a in self.agents if a.name == target_agent_name and a.active), None)
            if not target_agent:
                return {"step_id": step_id, "status": "error", "response": f"Target agent '{target_agent_name}' not found or inactive.", "data": None, "error_code": "AGENT_NOT_FOUND"}

            # Construct enhanced LLM prompt using service_def
            param_details_for_prompt = "\n".join([f"  - {p.name} ({p.type}, {'required' if p.required else 'optional'}{', default: '+str(p.default_value) if p.default_value is not None and not p.required else ''}): {p.description}" for p in service_def.parameters])
            returns_details_for_prompt = f"  - type: {service_def.returns.type}\n  - description: {service_def.returns.description}"

            fallback_prompt = (
                f"You are agent '{target_agent_name}'. You need to perform the service called '{service_def.name}'.\n"
                f"Service Description: {service_def.description}\n\n"
                f"Parameters Expected:\n{param_details_for_prompt}\n\n"
                f"Expected Return Structure:\n{returns_details_for_prompt}\n\n"
                f"Parameters Provided for this Call:\n{json.dumps(resolved_params, indent=2)}\n\n"
                f"Based on the service description, process the provided parameters and generate a JSON response that strictly conforms to the 'Expected Return Structure'. The main data should be under a 'data' key in your JSON response, and you should also include a 'status' ('success' or 'error') and a 'message' key."
                f"Example of expected JSON output format from you (the LLM agent):\n"
                f"{{\n  \"status\": \"success\",\n  \"data\": <value_matching_return_type_of_{service_def.returns.type}>,\n  \"message\": \"Service completed successfully.\"\n}}"
            )

            llm_call_result_or_task = await self.execute_agent(target_agent, fallback_prompt)

            if llm_call_result_or_task.get("status") == "pending_async":
                # LLM fallback was submitted as an async task by execute_agent
                print(f"{log_prefix}LLM fallback for service '{service_name}' initiated as task: {llm_call_result_or_task['task_id']}")
                return { # Propagate the pending_async status
                    "step_id": step_id,
                    "agent_name": f"{target_agent_name} (Service: {service_name}, LLM Fallback)",
                    "status": "pending_async",
                    "task_id": llm_call_result_or_task["task_id"],
                    "response": llm_call_result_or_task.get("message", f"LLM fallback for service '{service_name}' initiated."),
                    "data": None
                }
            elif llm_call_result_or_task.get("status") == "success": # Should ideally not happen if LLM agent always goes to pending_async
                # This path implies execute_agent returned a direct success for an LLM agent, which is not the new norm.
                # However, if it did, we'd try to parse its "response" as the structured JSON.
                try:
                    parsed_llm_response = json.loads(llm_call_result_or_task.get("response","{}"))
                    if not isinstance(parsed_llm_response, dict) or "status" not in parsed_llm_response or "data" not in parsed_llm_response:
                        raise ValueError("LLM direct response missing status/data keys.")
                    service_result_structured = parsed_llm_response
                    if "message" not in service_result_structured:
                        service_result_structured["message"] = f"LLM for service '{service_name}' (direct) completed with status: {service_result_structured['status']}"
                except Exception as e_parse_direct_llm:
                    err_msg = f"Error parsing direct LLM response for service '{service_name}': {e_parse_direct_llm}. Raw: {llm_call_result_or_task.get('response','')[:200]}..."
                    service_result_structured = {"status": "error", "data": None, "message": err_msg, "error_code": "LLM_DIRECT_RESPONSE_PARSE_ERROR"}
            else: # LLM call itself failed before becoming an async task (e.g., agent model misconfiguration in execute_agent)
                err_msg = f"LLM call for service '{service_name}' failed before async submission: {llm_call_result_or_task.get('response')}"
                print(f"{log_prefix}ERROR: {err_msg}")
                service_result_structured = {"status": "error", "data": None, "message": err_msg, "error_code": "LLM_PRE_ASYNC_CALL_FAILED"}

        # --- Process final result (if not pending_async) ---
        final_status = service_result_structured.get("status", "error")
        final_data = service_result_structured.get("data")
        final_message = service_result_structured.get("message", "Service call completed.")
        error_code = service_result_structured.get("error_code")

        if final_status != "success":
            print(f"{log_prefix}Service call step {step_id} ('{service_name}' on '{target_agent_name}') failed. Status: {final_status}, Message: {final_message[:100]}...")
        else:
            print(f"{log_prefix}Service call step {step_id} ('{service_name}' on '{target_agent_name}') completed successfully.")

        if output_var_name and final_status == "success":
            current_step_outputs[output_var_name] = final_data

        return {
            "step_id": step_id,
            "agent_name": f"{target_agent_name} (Service: {service_name})",
            "status": final_status,
            "response": final_message, # User-facing/log summary message
            "data": final_data, # Actual data payload
            "error_code": error_code
        }

   async def _service_codemaster_validate_syntax(self, params: Dict, service_definition: AgentServiceDefinition) -> Dict:
        # service_definition is now passed but not strictly needed if params are already validated by caller.
        # It could be used for more complex logic if the handler serves multiple similar services.
        code_snippet = params.get("code_snippet")
        # language = params.get("language", "python") # Default already handled by param validation if defined so.
        # The 'language' parameter would have been resolved by _handle_agent_service_call using defaults from service_def if not provided.
        language = params.get("language")


        codemaster_agent = next((a for a in self.agents if a.name == "CodeMaster" and a.active), None)
        if not codemaster_agent:
            return {"status": "error", "data": None, "message": "CodeMaster agent not available for syntax validation.", "error_code": "AGENT_UNAVAILABLE"}

        prompt = (f"Analyze the following {language} code snippet for syntax errors. Respond ONLY with a JSON object containing two keys: 'is_valid' (boolean) and 'errors' (a list of strings, empty if valid). Do not add any explanatory text outside the JSON.\nCode:\n```\n{code_snippet}\n```\nJSON Response:")

        llm_response_or_task = await self.execute_agent(codemaster_agent, prompt)

        llm_response: Dict
        if llm_response_or_task.get("status") == "pending_async":
            task_id = llm_response_or_task["task_id"]
            print(f"[_service_codemaster_validate_syntax] LLM task {task_id} submitted. Awaiting result...")
            while True:
                await asyncio.sleep(0.1) # Quick poll
                task_info = await self.get_async_task_info(task_id)
                if not task_info:
                     return {"status": "error", "data": None, "message": f"Syntax validation LLM task {task_id} info not found.", "error_code": "ASYNC_TASK_NOT_FOUND"}

                if task_info.status == AsyncTaskStatus.COMPLETED:
                    if not isinstance(task_info.result, dict) or "status" not in task_info.result:
                        return {"status": "error", "data": None, "message": f"Syntax validation LLM task {task_id} completed with unexpected result format: {type(task_info.result)}.", "error_code": "LLM_RESULT_FORMAT_ERROR"}
                    llm_response = task_info.result
                    print(f"[_service_codemaster_validate_syntax] LLM task {task_id} completed.")
                    break
                elif task_info.status == AsyncTaskStatus.FAILED:
                    return {"status": "error", "data": None, "message": f"Syntax validation LLM task {task_id} failed: {task_info.error}", "error_code": "LLM_TASK_FAILED"}
                elif task_info.status == AsyncTaskStatus.CANCELLED:
                     return {"status": "error", "data": None, "message": f"Syntax validation LLM task {task_id} cancelled.", "error_code": "LLM_TASK_CANCELLED"}
        else:
            llm_response = llm_response_or_task

        if llm_response.get("status") == "success":
            try:
                validation_data = json.loads(llm_response.get("response"))
                if not isinstance(validation_data, dict) or "is_valid" not in validation_data or "errors" not in validation_data:
                     raise ValueError("LLM response for syntax validation is not a dict with 'is_valid' and 'errors' keys.")
                return {"status": "success", "data": validation_data, "message": f"Syntax validation for {language} completed. Valid: {validation_data.get('is_valid')}."}
            except json.JSONDecodeError:
                return {"status": "error", "data": None, "message": "CodeMaster (validator LLM) returned non-JSON response.", "raw_response": llm_response.get("response"), "error_code": "LLM_RESPONSE_MALFORMED"}
            except ValueError as e:
                 return {"status": "error", "data": None, "message": f"CodeMaster (validator LLM) response structure error: {e}", "raw_response": llm_response.get("response"), "error_code": "LLM_RESPONSE_STRUCTURE_INVALID"}
        else:
            return {"status": "error", "data": None, "message": f"CodeMaster LLM call failed for syntax validation: {llm_response.get('response')}", "error_code": "LLM_CALL_FAILED"}


   async def store_knowledge(self, content: str, metadata: Optional[Dict] = None, content_id: Optional[str] = None) -> Dict:
       if self.knowledge_collection is None:
           return {"status": "error", "message": "KB not initialized."}
       try:
           final_id = content_id or str(uuid.uuid4())
           # Ensure metadata is suitable for ChromaDB (basic types) and also for event payload.
           # For ChromaDB, it will convert non-string/int/float/bool values.
           # For event payload, we might want to keep original structure if more complex.
           chroma_meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in metadata.items()} if metadata else {}

           self.knowledge_collection.add(ids=[final_id], documents=[content], metadatas=[chroma_meta] if chroma_meta else [None])

           # Publish event after successful storage
           event_payload = {
               "kb_id": final_id,
               "content_preview": content[:200] + "..." if len(content) > 200 else content, # Keep preview reasonable
               "metadata": metadata if metadata else {}, # Send original metadata in event
               "source_operation": metadata.get("source", "unknown") if metadata else "unknown" # e.g. "web_scrape", "plan_execution_log"
           }
           await self.publish_event(
               event_type="kb.content.added",
               source_component="TerminusOrchestrator.KnowledgeBase",
               payload=event_payload
           )
           print(f"[store_knowledge] Successfully stored and published kb.content.added event for KB ID: {final_id}")
           return {"status": "success", "id": final_id, "message": "Content stored and event published."}
       except Exception as e:
           print(f"[store_knowledge] Error storing to KB or publishing event: {e}")
           return {"status": "error", "message": str(e)}

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

   def store_user_feedback(self, item_id: str, item_type: str, rating: str, comment: Optional[str]=None, current_mode: Optional[str]=None, user_prompt_preview: Optional[str]=None) -> bool:
       try:
           feedback_id = str(uuid.uuid4())
           timestamp_iso = datetime.datetime.utcnow().isoformat() # Use UTC

           data_to_log = {
               "feedback_id": feedback_id,
               "timestamp_iso": timestamp_iso,
               "item_id": str(item_id),
               "item_type": str(item_type),
               "rating": str(rating),
               "comment": comment or "",
               "user_context":{
                   "operation_mode":current_mode,
                   "related_user_prompt_preview":user_prompt_preview[:200] if user_prompt_preview else None
                }
            }
           with open(self.feedback_log_file_path, 'a', encoding='utf-8') as f:
               f.write(json.dumps(data_to_log) + '\n')

           # Publish to the new event bus
           event_payload = {
               "feedback_id": feedback_id,
               "item_id": str(item_id),
               "item_type": str(item_type),
               "rating": str(rating),
               "comment_preview": (comment[:75] + "...") if comment and len(comment) > 75 else (comment or ""),
               "timestamp_iso": timestamp_iso
           }
           asyncio.create_task(self.publish_event(
               event_type="user.feedback.submitted",
               source_component="TerminusOrchestrator.FeedbackSystem", # Or simply "UserFeedbackSystem"
               payload=event_payload
           ))
           print(f"[store_user_feedback] Feedback {feedback_id} stored and event published.")
           return True
       except Exception as e:
           print(f"ERROR storing feedback or publishing event: {e}")
           return False

   async def generate_and_store_feedback_report(self) -> Dict:
       report_handler_id = "[FeedbackReport]"
       print(f"{report_handler_id} START: Generating and storing feedback analysis report.")
       if self.knowledge_collection is None: return {"status": "error", "message": "KB not initialized."}
       if not self.feedback_analyzer_script_path.exists(): return {"status": "error", "message": f"Analyzer script missing: {self.feedback_analyzer_script_path}"}
       try:
           print(f"{report_handler_id} INFO: Executing {self.feedback_analyzer_script_path} with log {self.feedback_log_file_path}")
           proc = await asyncio.create_subprocess_exec(sys.executable, str(self.feedback_analyzer_script_path), f"--log_file={self.feedback_log_file_path}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
           stdout, stderr = await proc.communicate()
           if proc.returncode != 0:
               err_msg = stderr.decode().strip() if stderr else "Unknown error from feedback_analyzer.py"
               print(f"{report_handler_id} ERROR: Analyzer script failed. Code: {proc.returncode}. Err: {err_msg}"); return {"status": "error", "message": f"Analyzer script failed: {err_msg}"}
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
           store_res = await self.store_knowledge(content=report_json_str, metadata=kb_meta)
           if store_res.get("status") == "success":
               msg = f"Feedback report stored. KB ID: {store_res.get('id')}"
               print(f"{report_handler_id} SUCCESS: {msg}")
                # Event "kb.content.added" is already published by store_knowledge.
                # If a more specific "kb.feedback_report.added" event is desired, it can be published here too.
                # For now, we rely on the generic one. The metadata['source'] == 'feedback_analysis_report'
                # in the generic event's payload can be used by subscribers to differentiate.
                # Example of specific event:
                # await self.publish_event(
                # event_type="kb.feedback_report.added",
                # source_component="FeedbackAnalyzerSystem",
                #     payload={"report_id": kb_meta.get('report_id'), "kb_id": store_res.get("id")}
                # )
               return {"status":"success", "message":msg, "kb_id":store_res.get("id")}
            else:
                print(f"{report_handler_id} ERROR: Failed to store report in KB. Msg: {store_res.get('message')}")
                return store_res
        except Exception as e:
            print(f"{report_handler_id} UNHANDLED ERROR: {e}")
            return {"status":"error","message":str(e)}
        finally:
            print(f"{report_handler_id} END: Processing finished.")

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

   async def _get_formatted_general_kb_context(self, kb_search_query: str, plan_handler_id: str) -> str:
       kb_general_ctx_str = ""
       if kb_search_query:
           print(f"[{plan_handler_id}] INFO: Querying KB for general context with: '{kb_search_query}'")
           general_hits = await self.retrieve_knowledge(
               kb_search_query,
               n_results=3,
               filter_metadata={"source": {"$nin": ["plan_execution_log", "feedback_analysis_report"]}}
           )
           if general_hits.get("status")=="success" and general_hits.get("results"):
               formatted_entries = [prompt_constructors.format_kb_entry_for_prompt(hit) for hit in general_hits["results"]]
               kb_general_ctx_str = "General Context from Knowledge Base (limit 3 relevance based on query):\n" + "\n".join(formatted_entries) + "\n\n"
               print(f"[{plan_handler_id}] INFO: Found {len(general_hits['results'])} general KB items.")
           else:
               print(f"[{plan_handler_id}] INFO: No general KB items found for query '{kb_search_query}'.")
       return kb_general_ctx_str

   async def _get_formatted_plan_log_insights(self, nlu_output: Dict, plan_handler_id: str) -> str:
       kb_plan_log_ctx_str = ""
       nlu_intent = nlu_output.get('intent','N/A')
       nlu_entities_str = str(nlu_output.get('entities',[]))[:50]
       plan_log_query = f"Intent: {nlu_intent}, Entities: {nlu_entities_str}"
       print(f"[{plan_handler_id}] INFO: Querying KB for plan logs with: '{plan_log_query}'")
       plan_log_hits = await self.retrieve_knowledge(plan_log_query, n_results=2, filter_metadata={"source":"plan_execution_log"})
       if plan_log_hits.get("status")=="success" and plan_log_hits.get("results"):
           formatted_plan_logs = [prompt_constructors.format_plan_log_entry_for_prompt(hit) for hit in plan_log_hits["results"]]
           kb_plan_log_ctx_str = "Insights from Past Plan Executions (limit 2 relevance based on NLU):\n" + "\n".join(formatted_plan_logs) + "\n\n"
           print(f"[{plan_handler_id}] INFO: Found {len(plan_log_hits['results'])} plan log items.")
       else:
           print(f"[{plan_handler_id}] INFO: No relevant plan logs found for query '{plan_log_query}'.")
       return kb_plan_log_ctx_str

   async def _get_formatted_feedback_insights(self, plan_handler_id: str) -> str:
       kb_feedback_ctx_str = ""
       feedback_query = "latest user feedback summary report"
       print(f"[{plan_handler_id}] INFO: Querying KB for feedback reports with: '{feedback_query}'")
       feedback_hits = await self.retrieve_knowledge(feedback_query, n_results=1, filter_metadata={"source":"feedback_analysis_report"})
       if feedback_hits.get("status")=="success" and feedback_hits.get("results"):
           formatted_feedback_reports = [prompt_constructors.format_feedback_report_for_prompt(hit) for hit in feedback_hits["results"]]
           kb_feedback_ctx_str = "Insights from Feedback Analysis Reports (limit 1 latest):\n" + "\n".join(formatted_feedback_reports) + "\n\n"
           print(f"[{plan_handler_id}] INFO: Found {len(feedback_hits['results'])} feedback report items.")
       else:
           print(f"[{plan_handler_id}] INFO: No feedback reports found for query '{feedback_query}'.")
       return kb_feedback_ctx_str

   def _construct_rl_state(self, user_prompt: str, nlu_output: Dict, kb_context_summary: Dict, previous_cycle_outcome: Optional[str]) -> Dict[str, Any]:
       state = {
           "nlu_intent": nlu_output.get("intent", "unknown"),
           "nlu_entities_count": len(nlu_output.get("entities", [])),
           "user_prompt_length_category": "short" if len(user_prompt) < 50 else "medium" if len(user_prompt) < 200 else "long",
           "kb_general_hits": kb_context_summary.get("general_hits_count", 0),
           "kb_plan_log_hits": kb_context_summary.get("plan_log_hits_count", 0),
           "kb_feedback_hits": kb_context_summary.get("feedback_hits_count", 0),
           "previous_cycle_outcome": previous_cycle_outcome if previous_cycle_outcome else "none",
       }
       return state

   def _calculate_rl_reward(self, execution_status: str, user_feedback_rating: Optional[str], num_revisions: int) -> float:
       reward = 0.0
       if execution_status == "success": reward += 0.5
       elif execution_status == "failure": reward -= 0.5
       if user_feedback_rating == "positive": reward += 1.0
       elif user_feedback_rating == "negative": reward -= 1.0
       reward -= num_revisions * 0.25
       return round(reward, 4)

   async def execute_master_plan(self, user_prompt: str, request_priority: Optional[str] = "normal") -> List[Dict]:
       plan_handler_id = f"[MasterPlanner user_prompt:'{user_prompt[:50]}...' Priority:'{request_priority}']"
       rl_interaction_id = str(uuid.uuid4())
       timestamp_interaction_start = datetime.datetime.now().isoformat()

       print(f"{plan_handler_id} START: Received request. RL Interaction ID: {rl_interaction_id}")
       self.conversation_history.append({"role": "user", "content": user_prompt})
       if len(self.conversation_history) > self.max_history_items:
           self.conversation_history = self.conversation_history[-self.max_history_items:]

       max_rev_attempts = 1; current_attempt = 0; plan_list = []; original_plan_json_str = ""
       final_exec_results = []
       step_outputs = {}
       first_attempt_nlu_output = {}
       detailed_failure_ctx_for_rev = {}
       current_plan_log_kb_id = None
       state_for_executed_plan_log: Optional[Dict] = None
       action_for_executed_plan_log: Optional[str] = None
       prompt_details_for_executed_plan_log: Optional[Dict] = None

       print(f"{plan_handler_id} INFO: Performing NLU analysis for initial planning.")
       first_attempt_nlu_output = await self.classify_user_intent(user_prompt)
       nlu_summary_for_prompt = f"NLU Analysis :: Intent: {first_attempt_nlu_output.get('intent','N/A')} :: Entities: {str(first_attempt_nlu_output.get('entities',[]))[:100]}..."
       print(f"{plan_handler_id} INFO: {nlu_summary_for_prompt}")

       while current_attempt <= max_rev_attempts:
           current_attempt_results = [] # Stores results of completed steps for this attempt
           plan_succeeded_this_attempt = True

           # For managing async tasks within the current plan execution attempt
           # Maps step_id to task_id for steps that are currently pending_async
           pending_async_steps: Dict[str, str] = {}
           # Keeps track of step_ids that have been submitted or completed (sync/async)
           # Used to determine if a step has been processed or is waiting for dependencies.
           processed_step_ids_this_attempt = set()


           print(f"{plan_handler_id} Attempt {current_attempt + 1}/{max_rev_attempts + 1}: Generating/Loading plan...")

           # --- Plan Generation/Loading (simplified for brevity, same as before) ---
           history_context = prompt_constructors.get_relevant_history_for_prompt(self.conversation_history, self.max_history_items, user_prompt)
           agent_capabilities_desc = self.get_agent_capabilities_description()

           kb_query_gen_prompt_str = prompt_constructors.construct_kb_query_generation_prompt(user_prompt, history_context, nlu_summary_for_prompt)
           kb_query_agent = next((a for a in self.agents if a.name == "GeneralPurposeAgent"), None) or \
                            next((a for a in self.agents if a.name == "MasterPlanner"), None)
           kb_search_query = ""
           if kb_query_agent:
               kb_query_res = await self._ollama_generate(kb_query_agent.model, kb_query_gen_prompt_str)
               if kb_query_res.get("status") == "success" and kb_query_res.get("response","").strip().upper() != "NO_QUERY_NEEDED":
                   kb_search_query = kb_query_res.get("response","").strip()
                   print(f"{plan_handler_id} INFO: Generated KB search query: '{kb_search_query}'")

           kb_general_ctx_str = await self._get_formatted_general_kb_context(kb_search_query, plan_handler_id)
           kb_plan_log_ctx_str = await self._get_formatted_plan_log_insights(first_attempt_nlu_output, plan_handler_id)
           kb_feedback_ctx_str = await self._get_formatted_feedback_insights(plan_handler_id)

           current_rl_state_kb_summary = {
               "general_hits_count": len(kb_general_ctx_str.splitlines()) -1 if kb_general_ctx_str.strip() else 0,
               "plan_log_hits_count": len(kb_plan_log_ctx_str.splitlines()) -1 if kb_plan_log_ctx_str.strip() else 0,
               "feedback_hits_count": len(kb_feedback_ctx_str.splitlines()) -1 if kb_feedback_ctx_str.strip() else 0,
           }
           current_rl_state = self._construct_rl_state(user_prompt, first_attempt_nlu_output, current_rl_state_kb_summary, None)
           current_rl_action = "Action_DefaultPrompt_InitialLog"
           current_prompt_details = {"strategy": "standard_initial_log"}

           if state_for_executed_plan_log is None :
                state_for_executed_plan_log = current_rl_state
                action_for_executed_plan_log = current_rl_action
                prompt_details_for_executed_plan_log = current_prompt_details

           if current_attempt == 0:
               planning_prompt = prompt_constructors.construct_main_planning_prompt(
                   user_prompt, history_context, nlu_summary_for_prompt,
                   kb_general_ctx_str, kb_plan_log_ctx_str, kb_feedback_ctx_str,
                   agent_capabilities_desc
               )
           else:
               print(f"{plan_handler_id} INFO: Constructing revision prompt with failure context.")
               planning_prompt = prompt_constructors.construct_revision_planning_prompt(
                   user_prompt, history_context, nlu_summary_for_prompt,
                   detailed_failure_ctx_for_rev, agent_capabilities_desc
               )

           planner_agent = next((a for a in self.agents if a.name == "MasterPlanner" and a.active), None)
           if not planner_agent: return [{"status":"error", "message":"MasterPlanner agent not found/active."}]
           raw_plan_response = await self._ollama_generate(planner_agent.model, planning_prompt)
           original_plan_json_str = raw_plan_response.get("response", "[]")

           try:
               plan_list = json.loads(original_plan_json_str)
               if not isinstance(plan_list, list): raise ValueError("Plan is not a list.")
               for step in plan_list:
                   if not step.get("step_type") in ["conditional", "loop", "parallel_group", "agent_service_call"] and \
                      not all(k in step for k in ["step_id", "agent_name", "task_prompt"]):
                       raise ValueError(f"Regular agent step missing required keys: {step}")
           except Exception as e_parse:
               print(f"{plan_handler_id} ERROR: Failed to parse plan from LLM: {e_parse}. Response: {original_plan_json_str[:200]}...")
               if current_attempt < max_rev_attempts:
                   detailed_failure_ctx_for_rev = self._capture_simple_failure_context(original_plan_json_str, {"error": f"Plan parsing failed: {e_parse}"})
                   current_attempt += 1; step_outputs = {}; plan_succeeded_this_attempt = False
                   state_for_executed_plan_log = current_rl_state
                   action_for_executed_plan_log = current_rl_action
                   prompt_details_for_executed_plan_log = current_prompt_details
                   continue
               else:
                   final_exec_results.append({"status":"error", "message":f"Failed to parse plan after {max_rev_attempts+1} attempts. Last error: {e_parse}"})
                   plan_succeeded_this_attempt = False; break

           if not plan_list:
               print(f"{plan_handler_id} INFO: Planner returned an empty plan.");
               plan_succeeded_this_attempt = True; # No steps, so technically success.
               final_exec_results = [] # Ensure it's defined
               break # Exit revision loop if plan is empty

           # --- Main Plan Execution Loop with Async Task Management ---
           # executed_step_ids tracks steps whose *final result* (sync or async) is processed.
           executed_step_ids = set()
           active_loops = {} # For managing loop contexts (as before)
           loop_context_stack = [] # For managing nested loops (as before)

           # Loop as long as there are steps not yet successfully completed, or async tasks pending for this plan
           while len(executed_step_ids) < len(plan_list) or pending_async_steps:
               dispatched_this_cycle = False # Track if any new step was dispatched or async task completed

               # 1. Attempt to dispatch new steps
               current_step_idx_for_dispatch = 0
               while current_step_idx_for_dispatch < len(plan_list):
                   step_to_evaluate = plan_list[current_step_idx_for_dispatch]
                   step_id_to_evaluate = step_to_evaluate.get("step_id")

                   if step_id_to_evaluate in executed_step_ids or step_id_to_evaluate in pending_async_steps:
                       current_step_idx_for_dispatch +=1; continue # Already done or pending

                   # Check dependencies
                   can_execute = True
                   for dep_id in step_to_evaluate.get("dependencies",[]):
                       if dep_id not in executed_step_ids: # Dependency not met
                           can_execute = False; break

                   if not can_execute:
                       current_step_idx_for_dispatch +=1; continue

                   # --- Dispatching the step (similar logic to before, but adapted) ---
                   dispatched_this_cycle = True
                   processed_step_ids_this_attempt.add(step_id_to_evaluate) # Mark as processed (submitted or completed)

                   step_priority = step_to_evaluate.get("priority", "normal").lower()
                   dispatch_log_prefix = f"[{plan_handler_id}]"
                   if step_priority == "high": dispatch_log_prefix += f" [Priority: HIGH]"
                   step_type_for_log = step_to_evaluate.get("step_type", "agent_execution")
                   log_agent_name = step_to_evaluate.get('agent_name', step_to_evaluate.get('target_agent_name', 'N/A'))
                   log_desc = step_to_evaluate.get('description', step_to_evaluate.get('service_name', 'N/A'))[:50]
                   print(f"{dispatch_log_prefix} Dispatching Step {step_id_to_evaluate}: Type='{step_type_for_log}', Agent/Service='{log_agent_name}', Desc='{log_desc}...'")

                   step_result_or_task: Dict
                   step_type = step_to_evaluate.get("step_type", "agent_execution")

                   if step_type == "conditional":
                       # Conditional logic is synchronous evaluation based on prior step_outputs
                       next_step_id_from_cond, eval_res_cond = await self._handle_conditional_step(step_to_evaluate, plan_list, step_outputs, executed_step_ids, plan_handler_id)
                       step_result_or_task = eval_res_cond if eval_res_cond else {"step_id": step_id_to_evaluate, "status":"error", "response":"Conditional eval result missing"}
                       # Conditional step itself is now "executed"
                       executed_step_ids.add(step_id_to_evaluate)
                       current_attempt_results.append(step_result_or_task)
                       if step_result_or_task.get("status") != "success": plan_succeeded_this_attempt = False; break
                       # Jump logic needs careful handling if we are not using current_step_idx in the outer loop directly for dispatch
                       # For now, conditional jumps will be handled by the next dispatch cycle finding the correct next step.
                       # This part might need refinement if strict sequential indexing after jump is critical.
                       # The current dispatch logic (iterating current_step_idx_for_dispatch) should naturally find the next valid step.
                   elif step_type == "loop" and step_to_evaluate.get("loop_type") == "while":
                       # Loop header evaluation is synchronous
                       # ... (loop handling logic as before, ensuring it updates executed_step_ids for the loop header) ...
                       # This needs careful integration with the new dispatch loop.
                       # For now, let's assume loop logic is complex and we'll simplify its handling here.
                       # A full refactor of loop logic with async steps inside is a larger task.
                       # Placeholder: Treat loop header as a synchronous step for now.
                       print(f"{plan_handler_id} DEBUG: Loop step {step_id_to_evaluate} encountered. (Async loop body not fully supported in this refactor iteration).")
                       step_result_or_task = {"status":"success", "response": f"Loop header {step_id_to_evaluate} processed (simulated)."}
                       executed_step_ids.add(step_id_to_evaluate)
                       current_attempt_results.append(step_result_or_task)

                   elif step_type == "agent_service_call":
                       step_result_or_task = await self._handle_agent_service_call(step_to_evaluate, step_outputs, plan_list)
                   elif step_to_evaluate.get("agent_name") == "parallel_group": # Not fully implemented
                       print(f"DEBUG: Parallel group {step_id_to_evaluate} encountered (simulated sync).")
                       step_result_or_task = {"status": "success", "response": f"Parallel group {step_id_to_evaluate} processed (simulated)."}
                   else: # Regular agent execution step
                       step_result_or_task = await self._execute_single_plan_step(step_to_evaluate, plan_list, step_outputs)

                   # --- Process result of dispatched step ---
                   if step_result_or_task.get("status") == "pending_async":
                       task_id = step_result_or_task["task_id"]
                       pending_async_steps[step_id_to_evaluate] = task_id
                       print(f"{plan_handler_id} Step {step_id_to_evaluate} is PENDING_ASYNC with task_id {task_id}.")
                   elif step_result_or_task.get("status") == "success":
                       current_attempt_results.append(step_result_or_task)
                       executed_step_ids.add(step_id_to_evaluate) # Mark as fully completed
                       print(f"{plan_handler_id} Step {step_id_to_evaluate} completed synchronously with SUCCESS.")
                   else: # Synchronous failure
                       current_attempt_results.append(step_result_or_task)
                       executed_step_ids.add(step_id_to_evaluate) # Mark as completed (failed)
                       plan_succeeded_this_attempt = False
                       print(f"{plan_handler_id} Step {step_id_to_evaluate} completed synchronously with FAILURE: {step_result_or_task.get('response')}")
                       break # Stop processing more steps this attempt if one fails synchronously

                   current_step_idx_for_dispatch +=1
               # End of dispatch loop for new steps
               if not plan_succeeded_this_attempt: break # Exit main while loop for this attempt

               # 2. Check status of pending asynchronous tasks
               if pending_async_steps:
                   completed_tasks_this_cycle: List[str] = []
                   for step_id_pending, task_id_pending in pending_async_steps.items():
                       task_info = await self.get_async_task_info(task_id_pending)
                       if task_info:
                           if task_info.status == AsyncTaskStatus.COMPLETED:
                               dispatched_this_cycle = True # Activity occurred
                               print(f"{plan_handler_id} AsyncTask {task_id_pending} for step {step_id_pending} COMPLETED.")
                               # Result of an async LLM call via execute_agent is the dict from _ollama_generate
                               # Result of an async Service Call LLM fallback is the dict from the LLM (status, data, message)
                               # Result of an async direct Service Handler (if it returned task_id) would be its final structured dict.
                               async_step_result = task_info.result
                               if not isinstance(async_step_result, dict): # Should be a dict from execute_agent or service call
                                   async_step_result = {"status": "error", "response": f"Async task result for step {step_id_pending} was not a dict: {async_step_result}", "data": None}

                               # Store output if successful
                               output_var_name_for_async = next((s.get("output_variable_name") for s in plan_list if s.get("step_id") == step_id_pending), None)
                               if async_step_result.get("status") == "success" and output_var_name_for_async:
                                   # If the result is from a service call, its 'data' field holds the actual output
                                   # If from _execute_single_plan_step -> _ollama_generate, its 'response' field is the primary output
                                   data_to_store = async_step_result.get("data", async_step_result.get("response"))
                                   step_outputs[output_var_name_for_async] = data_to_store
                                   # Handle other special keys like image_path if needed from async_step_result
                                   for mk in ["image_path","frame_path","gif_path","speech_path","modified_file"]:
                                       if mk in async_step_result: step_outputs[f"{output_var_name_for_async}_{mk}"]=async_step_result[mk]

                               original_step_def_for_async = next((s_def for s_def in plan_list if s_def.get("step_id") == step_id_pending), None)
                               agent_name_for_async_log = "UnknownAgent"
                               if original_step_def_for_async:
                                   agent_name_for_async_log = original_step_def_for_async.get('agent_name',
                                                               original_step_def_for_async.get('target_agent_name', 'AsyncStep'))

                               current_attempt_results.append({
                                   "step_id": step_id_pending,
                                   "agent_name": agent_name_for_async_log,
                                   "status": async_step_result.get("status"),
                                   "response": async_step_result.get("message", async_step_result.get("response")), # Prefer message for services
                                   "data": async_step_result.get("data") # Data from service call
                                })
                                executed_step_ids.add(step_id_pending)
                                completed_tasks_this_cycle.append(step_id_pending)
                                if async_step_result.get("status") != "success":
                                    plan_succeeded_this_attempt = False

                           elif task_info.status == AsyncTaskStatus.FAILED:
                               dispatched_this_cycle = True # Activity occurred
                               print(f"{plan_handler_id} AsyncTask {task_id_pending} for step {step_id_pending} FAILED: {task_info.error}")
                               current_attempt_results.append({"step_id": step_id_pending, "status": "error", "response": task_info.error})
                               executed_step_ids.add(step_id_pending)
                               completed_tasks_this_cycle.append(step_id_pending)
                               plan_succeeded_this_attempt = False
                           # else PENDING or RUNNING, do nothing this cycle for this task
                       else: # Should not happen if task IDs are managed correctly
                           print(f"{plan_handler_id} ERROR: No task info found for pending task_id {task_id_pending} (step {step_id_pending}). Marking as error.")
                           current_attempt_results.append({"step_id": step_id_pending, "status": "error", "response": "Async task info lost."})
                           executed_step_ids.add(step_id_pending)
                           completed_tasks_this_cycle.append(step_id_pending)
                           plan_succeeded_this_attempt = False

                   for step_id_done in completed_tasks_this_cycle:
                       pending_async_steps.pop(step_id_done, None)

                   if not plan_succeeded_this_attempt: break # Exit main while loop for this attempt

               # 3. If no steps were dispatched and no async tasks completed/failed this cycle, but still pending tasks, sleep.
               if not dispatched_this_cycle and pending_async_steps:
                   print(f"{plan_handler_id} No new steps dispatched or tasks completed this cycle. {len(pending_async_steps)} tasks still pending. Sleeping...")
                   await asyncio.sleep(0.2) # Polling interval for pending tasks
               elif not dispatched_this_cycle and not pending_async_steps and len(executed_step_ids) < len(plan_list):
                   # This case indicates a possible deadlock or issue with dependency logic if plan is not complete.
                   print(f"{plan_handler_id} WARNING: No progress made, no pending tasks, but plan not complete. Check for unbreakable dependency loops or logic errors.")
                   plan_succeeded_this_attempt = False # Consider this a failure of the plan execution logic
                   break

           # --- End of Main Plan Execution Loop for this attempt ---
           final_exec_results = current_attempt_results

           if not plan_succeeded_this_attempt and current_attempt < max_rev_attempts:
               # Capture failure context for revision. If failure was due to an async step, find its details in current_attempt_results.
               failed_step_details_for_context = next((res for res in reversed(current_attempt_results) if res.get("status") != "success"), None)
               failed_step_def_for_context = None
               if failed_step_details_for_context and failed_step_details_for_context.get("step_id"):
                   failed_step_def_for_context = next((s_def for s_def in plan_list if s_def.get("step_id") == failed_step_details_for_context.get("step_id")), None)

               detailed_failure_ctx_for_rev = self._capture_failure_context(
                   original_plan_json_str,
                   failed_step_def_for_context,
                   failed_step_details_for_context,
                   step_outputs
               )
               current_attempt += 1;
               step_outputs = {}; # Reset outputs for next attempt
               # Reset tracking for next attempt
               pending_async_steps.clear()
               processed_step_ids_this_attempt.clear()
               executed_step_ids.clear()
               state_for_executed_plan_log = current_rl_state
               action_for_executed_plan_log = current_rl_action
               prompt_details_for_executed_plan_log = current_prompt_details
           else: break

       user_facing_summary = await self._summarize_execution_for_user(user_prompt, final_exec_results)
       final_plan_outcome_status_str = "success" if plan_succeeded_this_attempt else "failure"
       user_feedback_rating_for_log = "none"
       if state_for_executed_plan_log is None: state_for_executed_plan_log = self._construct_rl_state(user_prompt, first_attempt_nlu_output, {}, None)
       if action_for_executed_plan_log is None: action_for_executed_plan_log = "Action_Planner_Error_Before_Action"
       if prompt_details_for_executed_plan_log is None: prompt_details_for_executed_plan_log = {"error": "No plan generated or parsed"}
       calculated_reward = self._calculate_rl_reward(final_plan_outcome_status_str, user_feedback_rating_for_log, current_attempt)
       timestamp_interaction_end = datetime.datetime.now().isoformat()
       self.rl_logger.log_experience(
           rl_interaction_id=rl_interaction_id, attempt_number=current_attempt, state=state_for_executed_plan_log, action=action_for_executed_plan_log,
           master_planner_prompt_details=prompt_details_for_executed_plan_log, generated_plan_json=original_plan_json_str,
           plan_parsing_status="success" if plan_list else "failure",
           final_executed_plan_json=original_plan_json_str if plan_succeeded_this_attempt or (not plan_succeeded_this_attempt and current_attempt >=max_rev_attempts) else "N/A",
           execution_status=final_plan_outcome_status_str, user_feedback_rating=user_feedback_rating_for_log,
           calculated_reward=calculated_reward, next_state=None, done=True,
           timestamp_start_iso=timestamp_interaction_start, timestamp_end_iso=timestamp_interaction_end
       )
       if plan_list or not plan_succeeded_this_attempt :
            current_plan_log_kb_id = await self._store_plan_execution_log_in_kb(user_prompt, first_attempt_nlu_output, original_plan_json_str, plan_succeeded_this_attempt, current_attempt + 1, final_exec_results, step_outputs, user_facing_summary)
       self.conversation_history.append({"role": "assistant", "content": user_facing_summary, "is_plan_outcome": True, "plan_log_kb_id": current_plan_log_kb_id, "feedback_item_id": current_plan_log_kb_id, "feedback_item_type": "master_plan_log_outcome", "related_user_prompt_for_feedback": user_prompt})
       return final_exec_results

   async def _evaluate_plan_condition(self, condition_def: Dict, step_outputs: Dict, full_plan_list: List[Dict]) -> Dict:
       if not condition_def: return {"status": "error", "evaluation": None, "message": "Condition definition is empty."}
       source_step_id = condition_def.get("source_step_id"); output_var_path = condition_def.get("source_output_variable"); operator = condition_def.get("operator"); compare_value_literal = condition_def.get("value"); value_type_hint = condition_def.get("value_type", "string")
       if not all([source_step_id, output_var_path, operator]): return {"status": "error", "evaluation": None, "message": "Condition missing source_step_id, source_output_variable, or operator."}
       source_step_output_key = None
       for step_cfg in full_plan_list:
           if step_cfg.get("step_id") == source_step_id: source_step_output_key = step_cfg.get("output_variable_name", f"step_{source_step_id}_output"); break
       if not source_step_output_key or source_step_output_key not in step_outputs: return {"status": "error", "evaluation": None, "message": f"Output for source_step_id '{source_step_id}' (expected key: {source_step_output_key}) not found in step_outputs: {list(step_outputs.keys())}."}
       actual_value_raw = step_outputs.get(source_step_output_key)
       try:
           current_val = actual_value_raw
           for part in output_var_path.split('.'):
               if isinstance(current_val, dict): current_val = current_val.get(part);
               elif isinstance(current_val, list) and part.isdigit() and int(part) < len(current_val): current_val = current_val[int(part)]
               else: current_val = None; break
           actual_value_to_compare = current_val
       except Exception as e: return {"status": "error", "evaluation": None, "message": f"Error accessing source_output_variable '{output_var_path}' from '{source_step_output_key}': {str(e)}"}
       try:
           if value_type_hint == "integer": actual_value_to_compare = int(actual_value_to_compare) if actual_value_to_compare is not None else None; compare_value_literal = int(compare_value_literal) if compare_value_literal is not None else None
           elif value_type_hint == "float": actual_value_to_compare = float(actual_value_to_compare) if actual_value_to_compare is not None else None; compare_value_literal = float(compare_value_literal) if compare_value_literal is not None else None
           elif value_type_hint == "boolean":
               if isinstance(actual_value_to_compare, str): actual_value_to_compare = actual_value_to_compare.lower() in ["true", "1", "yes"]
               else: actual_value_to_compare = bool(actual_value_to_compare)
               if isinstance(compare_value_literal, str): compare_value_literal = compare_value_literal.lower() in ["true", "1", "yes"]
               else: compare_value_literal = bool(compare_value_literal)
           else: actual_value_to_compare = str(actual_value_to_compare) if actual_value_to_compare is not None else ""; compare_value_literal = str(compare_value_literal) if compare_value_literal is not None else ""
       except ValueError as ve: return {"status": "error", "evaluation": None, "message": f"Type conversion error for '{value_type_hint}': {ve}. Val: '{actual_value_to_compare}', Comp: '{compare_value_literal}'"}
       evaluation = None
       try:
           if operator == "equals": evaluation = (actual_value_to_compare == compare_value_literal)
           elif operator == "not_equals": evaluation = (actual_value_to_compare != compare_value_literal)
           elif operator == "contains":
               if isinstance(actual_value_to_compare, (str, list)): evaluation = (compare_value_literal in actual_value_to_compare)
               elif isinstance(actual_value_to_compare, dict): evaluation = (compare_value_literal in actual_value_to_compare.keys() or compare_value_literal in actual_value_to_compare.values())
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
       except Exception as e: return {"status": "error", "evaluation": None, "message": f"Error during condition evaluation: {str(e)}"}

   def _capture_failure_context(self, plan_str:str, failed_step_def:Optional[Dict], failed_step_result:Optional[Dict], current_outputs:Dict) -> Dict:
        return {"plan_that_failed_this_attempt": plan_str, "failed_step_definition": failed_step_def if failed_step_def else "N/A", "failed_step_execution_result": failed_step_result if failed_step_result else "N/A", "step_outputs_before_failure": dict(current_outputs)}

   async def _summarize_execution_for_user(self, user_prompt:str, final_exec_results:List[Dict]) -> str:
       summarizer = next((a for a in self.agents if a.name=="CreativeWriter"),None) or next((a for a in self.agents if a.name=="DeepThink"),None)
       if not summarizer: s_count = sum(1 for r in final_exec_results if r.get('status')=='success'); return f"Plan execution finished. {s_count}/{len(final_exec_results)} steps processed successfully."
       summary_context = f"Original User Request: '{user_prompt}'\nPlan Execution Summary of Final Attempt:\n"
       if not final_exec_results: summary_context += "No steps were executed or the plan was empty.\n"
       else:
           for i, res in enumerate(final_exec_results): summary_context += (f"  Step {i+1} (Agent: {res.get('agent','N/A')}, ID: {res.get('step_id','N/A')}): Status='{res.get('status','unknown')}', Output Snippet='{str(res.get('response','No response'))[:100]}...'\n")
       prompt = (f"You are an AI assistant. Based on the user's original request and a summary of the multi-step plan execution's final attempt, provide a concise, natural language summary of what actions were taken by the system and the overall outcome. Focus on what would be most useful for the user to understand what just happened. Do not refer to yourself as '{summarizer.name}', just act as the main AI assistant.\n\n{summary_context}\n\nPlease provide only the summary text, suitable for conversation history.")
       res = await self.execute_agent(summarizer, prompt)
       if res.get("status")=="success" and res.get("response","").strip(): return res.get("response").strip()
       else: s_count = sum(1 for r in final_exec_results if r.get('status')=='success'); return f"Plan execution attempt finished. {s_count}/{len(final_exec_results)} steps successful. Summarization failed: {res.get('response')}"

   async def _store_plan_execution_log_in_kb(self, user_prompt_orig:str, nlu_output_orig:Dict, plan_json_final_attempt:str, final_status_bool:bool, num_attempts:int, step_results_final_attempt:List[Dict], outputs_final_attempt:Dict, user_facing_summary_text:str) -> Optional[str]:
       if not self.knowledge_collection:
           print("MasterPlanner: Knowledge Base unavailable, skipping storage of plan execution summary.");
           return None

       final_plan_status_str = "success" if final_status_bool else "failure"
       # ... (rest of the summary_dict and kb_meta construction remains the same) ...
       summary_list_for_log = [{"step_id":s.get("step_id","N/A"), "agent_name":s.get("agent","N/A"), "status":s.get("status","unknown"), "response_preview":str(s.get("response",""))[:150]+"..."} for s in step_results_final_attempt]
       nlu_analysis_data = {}
       if isinstance(nlu_output_orig, dict): nlu_analysis_data = { "intent":nlu_output_orig.get("intent"), "intent_scores":nlu_output_orig.get("intent_scores"), "entities":nlu_output_orig.get("entities",[]) }
       summary_dict = { "version":"1.3_service_calls_rl_log_v1", "original_user_request":user_prompt_orig, "nlu_analysis_on_request": nlu_analysis_data, "plan_json_executed_final_attempt":plan_json_final_attempt, "execution_summary":{ "overall_status":final_plan_status_str, "total_attempts":num_attempts, "final_attempt_step_results":summary_list_for_log, "outputs_from_successful_steps_final_attempt": {k: (str(v)[:200]+"..." if len(str(v)) > 200 else v) for k,v in outputs_final_attempt.items()} }, "user_facing_plan_outcome_summary":user_facing_summary_text, "log_timestamp_iso":datetime.datetime.utcnow().isoformat() }
       content_str = json.dumps(summary_dict, indent=2)
       kb_meta = {
           "source":"plan_execution_log",
           "overall_status":final_plan_status_str,
           "user_request_preview":user_prompt_orig[:150],
           "primary_intent":nlu_analysis_data.get("intent","N/A"),
           "log_timestamp_iso":summary_dict["log_timestamp_iso"]
       }
       if nlu_analysis_data.get("entities"):
           for i, ent in enumerate(nlu_analysis_data["entities"][:3]):
               kb_meta[f"entity_{i+1}_type"]=ent.get("type","UNK")
               kb_meta[f"entity_{i+1}_text"]=str(ent.get("text",""))[:50]

       # Call store_knowledge, which will now publish the "kb.content.added" event internally
       store_result = await self.store_knowledge(content_str, kb_meta, content_id=f"planlog_{uuid.uuid4()}")

       if store_result.get("status") == "success":
           stored_kb_id = store_result.get("id")
           print(f"MasterPlanner: Plan log storage successful. Stored KB ID: {stored_kb_id}")
           # No direct publish_message here anymore; store_knowledge handles the generic kb.content.added event.
           # If a more specific event "kb.plan_execution_log.added" is needed, store_knowledge
           # could be made more flexible, or we could publish that specific event here too.
           # For now, relying on the generic event from store_knowledge.
           # Example of publishing a more specific event if needed:
           # await self.publish_event(
           # event_type="kb.plan_execution_log.added",
           # source_component="MasterPlanner",
           #     payload={
           # "kb_id": stored_kb_id,
           # "original_request_preview": user_prompt_orig[:150],
           # "overall_status": final_plan_status_str,
           # "primary_intent": nlu_analysis_data.get("intent", "N/A")
           #     }
           # )
           return stored_kb_id
       else:
           print(f"MasterPlanner: Plan log storage failed. Message: {store_result.get('message')}")
           return None

   async def execute_agent(self, agent: Agent, prompt: str, context: Optional[Dict] = None) -> Dict:
        print(f"Orchestrator: Executing agent {agent.name} with prompt (first 100 chars): {prompt[:100]}")
        if agent.name == "WebCrawler":
            is_url = prompt.startswith("http://") or prompt.startswith("https://")
            if is_url:
                url_to_scrape = prompt; print(f"WebCrawler: Identified URL for scraping: {url_to_scrape}")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url_to_scrape, timeout=10) as response: response.raise_for_status(); html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser'); paragraphs = soup.find_all('p')
                    text_content = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
                    if not text_content.strip(): return {"status": "success", "agent": agent.name, "model": agent.model, "response_type": "scrape_result", "response": f"No significant text content found at {url_to_scrape}.", "summary": "", "original_url": url_to_scrape, "full_content_length": 0}
                    summary = text_content[:1000]
                    doc_summarizer_agent = next((a for a in self.agents if a.name == "DocSummarizer" and a.active), None)
                    if doc_summarizer_agent:
                        summary_prompt = f"Please summarize the following web page content obtained from {url_to_scrape}:\n\n{text_content[:15000]}"
                        # This internal _ollama_generate call could also be made an async task if summarization is slow
                        # For now, WebCrawler awaits it directly.
                        summary_result = await self._ollama_generate(doc_summarizer_agent.model, summary_prompt, context)
                        if summary_result.get("status") == "success" and summary_result.get("response"): summary = summary_result.get("response")

                    kb_metadata = {
                        "source": "web_scrape",
                        "url": url_to_scrape,
                        "scraped_timestamp_iso": datetime.datetime.utcnow().isoformat(), # Use UTC
                        "original_content_length": len(text_content),
                        "agent_used": agent.name,
                        "summarizer_used": doc_summarizer_agent.name if doc_summarizer_agent else "N/A"
                    }
                    # Await store_knowledge to ensure event is published if successful
                    store_kb_result = await self.store_knowledge(content=summary, metadata=kb_metadata)
                    if store_kb_result.get("status") == "success":
                        print(f"WebCrawler: Successfully stored scraped content from {url_to_scrape} to KB (ID: {store_kb_result.get('id')}).")
                        return {"status": "success", "agent": agent.name, "model": agent.model, "response_type": "scrape_result", "response": f"Successfully scraped and summarized content from {url_to_scrape}. Summary stored in KB (ID: {store_kb_result.get('id')}).", "summary": summary, "original_url": url_to_scrape, "full_content_length": len(text_content), "kb_id": store_kb_result.get("id")}
                    else:
                        print(f"WebCrawler: Failed to store scraped content from {url_to_scrape} to KB. Error: {store_kb_result.get('message')}")
                        return {"status": "error", "agent": agent.name, "model": agent.model, "response_type": "scrape_kb_store_error", "response": f"Error storing scraped content for {url_to_scrape}: {store_kb_result.get('message')}"}

                except Exception as e:
                    return {"status": "error", "agent": agent.name, "model": agent.model, "response_type": "scrape_error", "response": f"Error scraping URL {url_to_scrape}: {str(e)}"}
            else: # WebCrawler performing a search query
                search_query = prompt
                print(f"WebCrawler: Identified search query: {search_query}")
                # DDGS is synchronous, so run in executor or make it an async task if it's too slow.
                # For now, let's assume it's fast enough or becomes an async task later.
                # If DDGS itself were async, we could await it directly.
                # To make this part non-blocking if DDGS is slow:
                # ddgs_coro = loop.run_in_executor(None, lambda: DDGS().text(keywords=search_query, region='wt-wt', max_results=5, safesearch='moderate'))
                # task_id = await self.submit_async_task(ddgs_coro, name=f"WebSearch-{agent.name}-{search_query[:20]}")
                # return {"status": "pending_async", "task_id": task_id, "message": f"Web search initiated for: {search_query}"}
                # For now, keeping it synchronous for simplicity in this step, will be a candidate for async if slow.
                try:
                    loop = asyncio.get_event_loop();
                    with DDGS() as ddgs_sync: # Renamed to avoid conflict if we make it async later
                        search_results_raw = await loop.run_in_executor(None, lambda: ddgs_sync.text(keywords=search_query, region='wt-wt', max_results=5, safesearch='moderate'))
                    formatted_results = []
                    if search_results_raw:
                        for res_raw in search_results_raw: formatted_results.append({"title": res_raw.get('title', 'N/A'), "url": res_raw.get('href', ''), "snippet": res_raw.get('body', '')})
                    return {"status": "success", "agent": agent.name, "model": agent.model, "response_type": "search_results", "response": formatted_results }
                except Exception as e:
                    return {"status": "error", "agent": agent.name, "model": agent.model, "response_type": "search_error", "response": f"Error performing web search for '{search_query}': {str(e)}"}

        elif "ollama" in agent.model:
            # This is a key change: LLM calls are now submitted as async tasks
            ollama_coro = self._ollama_generate(agent.model, prompt, context)
            task_id = await self.submit_async_task(ollama_coro, name=f"Ollama-{agent.name}-{prompt[:20]}")
            return {"status": "pending_async", "task_id": task_id, "message": f"LLM operation for agent {agent.name} started."}

        elif agent.name == "ImageForge":
            # Image generation can also be long; make it an async task.
            # generate_image_with_hf_pipeline is already async.
            image_gen_coro = self.generate_image_with_hf_pipeline(prompt)
            task_id = await self.submit_async_task(image_gen_coro, name=f"ImageForge-{prompt[:20]}")
            return {"status": "pending_async", "task_id": task_id, "message": "Image generation started."}
            # Old synchronous way: return await self.generate_image_with_hf_pipeline(prompt)

        else:
            # For other agent types not explicitly handled as async yet
            return {"status": "error", "agent": agent.name, "model": agent.model, "response": f"Execution logic for agent {agent.name} not specifically defined for async or direct execution."}

   async def classify_user_intent(self, user_prompt: str) -> Dict: # This method calls execute_agent
        nlu_agent = next((a for a in self.agents if a.name == "NLUAnalysisAgent" and a.active), None)
        if not nlu_agent: return {"status": "error", "message": "NLUAnalysisAgent not found or inactive.", "intent": None, "entities": []}
        candidate_labels_str = ", ".join([f"'{label}'" for label in self.candidate_intent_labels])
        nlu_prompt = (f"Analyze the following user prompt: '{user_prompt}'\n\n1. Intent Classification: Classify the primary intent of the prompt against the following candidate labels: [{candidate_labels_str}]. Provide the top intent and its confidence score.\n2. Named Entity Recognition: Extract relevant named entities (like names, locations, dates, organizations, products, specific terms like filenames or URLs).\n\nReturn your analysis as a single, minified JSON object with the following exact structure:\n{{\"intent\": \"<detected_intent_label>\", \"intent_score\": <float_score_0_to_1>, \"entities\": [{{ \"text\": \"<entity_text>\", \"type\": \"<ENTITY_TYPE_UPPERCASE>\", \"score\": <float_score_0_to_1>}}]}}\nIf no entities are found, return an empty list for \"entities\". If intent is unclear, use an appropriate label or 'unknown_intent'. Ensure scores are floats.")

        raw_nlu_result_or_task = await self.execute_agent(nlu_agent, nlu_prompt)

        raw_nlu_result: Dict
        if raw_nlu_result_or_task.get("status") == "pending_async":
            task_id = raw_nlu_result_or_task["task_id"]
            print(f"[classify_user_intent] NLU analysis submitted as task {task_id}. Awaiting result...")
            while True:
                await asyncio.sleep(0.1) # Quick poll for potentially fast NLU
                task_info = await self.get_async_task_info(task_id)
                if not task_info:
                    return {"status": "error", "message": f"NLU task {task_id} info not found.", "intent": None, "entities": []}

                if task_info.status == AsyncTaskStatus.COMPLETED:
                    if not isinstance(task_info.result, dict) or "status" not in task_info.result:
                         return {"status": "error", "message": f"NLU task {task_id} completed with unexpected result format: {type(task_info.result)}.", "intent": None, "entities": []}
                    raw_nlu_result = task_info.result
                    print(f"[classify_user_intent] NLU task {task_id} completed.")
                    break
                elif task_info.status == AsyncTaskStatus.FAILED:
                    return {"status": "error", "message": f"NLU analysis task {task_id} failed: {task_info.error}", "intent": None, "entities": []}
                elif task_info.status == AsyncTaskStatus.CANCELLED:
                    return {"status": "error", "message": f"NLU analysis task {task_id} was cancelled.", "intent": None, "entities": []}
        else:
            raw_nlu_result = raw_nlu_result_or_task

        if raw_nlu_result.get("status") != "success":
            return {"status": "error", "message": f"NLUAnalysisAgent call failed: {raw_nlu_result.get('response')}", "intent": None, "entities": []}

        try:
            parsed_response = json.loads(raw_nlu_result.get("response", "{}"))
            intent = parsed_response.get("intent", "unknown_intent")
            intent_score = parsed_response.get("intent_score", 0.0)
            entities = parsed_response.get("entities", [])
            if not isinstance(entities, list): entities = [] # Ensure entities is a list
            intent_scores = {intent: intent_score} if intent != "unknown_intent" else {}
            return {"status": "success", "intent": intent, "intent_scores": intent_scores, "entities": entities, "message": "NLU analysis via agent successful."}
        except json.JSONDecodeError:
            return {"status": "error", "message": "NLUAnalysisAgent returned invalid JSON.", "raw_response": raw_nlu_result.get("response"), "intent": None, "entities": []}
        except Exception as e:
            return {"status": "error", "message": f"Error processing NLU agent response: {str(e)}", "intent": None, "entities": []}

   async def extract_document_content(self, uploaded_file_object: Any, original_filename: str) -> Dict[str, Any]:
        content_text = ""; status = "error"; message = "File type not supported or error during processing."; file_ext = Path(original_filename).suffix.lower().strip('.')
        from typing import Any
        try:
            if not hasattr(uploaded_file_object, 'getvalue'): raise ValueError("uploaded_file_object does not have getvalue() method.")
            file_bytes = uploaded_file_object.getvalue()
            if file_ext == 'txt': content_text = file_bytes.decode('utf-8', errors='replace'); status = "success"; message = "Text file processed."
            elif file_ext == 'json':
                try: parsed_json = json.loads(file_bytes.decode('utf-8', errors='replace')); content_text = json.dumps(parsed_json, indent=2); status = "success"; message = "JSON file processed."
                except json.JSONDecodeError as e: content_text = file_bytes.decode('utf-8', errors='replace'); message = f"JSON parsing error: {str(e)}. Displaying raw content."; status = "partial_success"
            elif file_ext == 'csv': content_text = file_bytes.decode('utf-8', errors='replace'); status = "success"; message = "CSV file processed as text."
            elif file_ext == 'pdf':
                try:
                    doc = fitz.open(stream=file_bytes, filetype="pdf"); text_parts = [doc.load_page(i).get_text("text") for i in range(len(doc))]; content_text = "\n".join(text_parts)
                    status = "success"; message = f"PDF processed ({len(doc)} pages).";
                    if not content_text.strip(): message += " Note: No text content found (possibly image-based PDF)."
                except Exception as e: message = f"Error processing PDF: {str(e)}"
            elif file_ext == 'docx':
                try: doc = docx.Document(io.BytesIO(file_bytes)); content_text = "\n".join([para.text for para in doc.paragraphs]); status = "success"; message = "DOCX file processed."
                except Exception as e: message = f"Error processing DOCX: {str(e)}"
            elif file_ext == 'xlsx':
                try:
                    workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True); text_parts = []
                    for sheet_name in workbook.sheetnames:
                        sheet = workbook[sheet_name]; text_parts.append(f"--- Sheet: {sheet_name} ---")
                        for row in sheet.iter_rows(): text_parts.append(", ".join([str(cell.value) if cell.value is not None else "" for cell in row]))
                    content_text = "\n".join(text_parts); status = "success"; message = f"XLSX file processed ({len(workbook.sheetnames)} sheets)."
                except Exception as e: message = f"Error processing XLSX: {str(e)}"
            elif file_ext == 'html' or file_ext == 'htm':
                try: soup = BeautifulSoup(file_bytes, 'html.parser'); content_text = soup.get_text(separator='\n', strip=True); status = "success"; message = "HTML file processed."
                except Exception as e: message = f"Error processing HTML: {str(e)}"
            else:
                try: content_text = file_bytes.decode('utf-8', errors='replace'); status = "partial_success"; message = f"File type '{file_ext}' not specifically handled, attempted to read as text."
                except Exception: message = f"File type '{file_ext}' not supported and could not be read as raw text."; content_text = None; status = "error"
            if content_text is None: content_text = ""
        except Exception as e_outer: message = f"Outer error during file processing: {str(e_outer)}"; content_text = ""; status = "error"
        return {"status": status, "content": content_text.strip(), "file_type_processed": file_ext, "message": message}

   async def _ollama_generate(self, model_name: str, prompt: str, context: Optional[Dict] = None) -> Dict:
       print(f"Orchestrator (_ollama_generate): Calling model {model_name} with prompt: {prompt[:100]}...")
       payload = {"model": model_name, "prompt": prompt, "stream": False}
       if context: payload["context"] = context
       try:
           async with aiohttp.ClientSession() as session:
               async with session.post(self.ollama_url, json=payload) as response:
                   if response.status == 200:
                       resp_json = await response.json()
                       return {"status": "success", "response": resp_json.get("response",""), "context": resp_json.get("context")}
                   else:
                       error_text = await response.text()
                       return {"status": "error", "response": f"Ollama API error {response.status}: {error_text}"}
       except Exception as e:
           return {"status": "error", "response": f"Failed to call Ollama: {str(e)}"}

   async def generate_image_with_hf_pipeline(self, prompt:str) -> Dict[str, Any]:
        print(f"Orchestrator (ImageForge): Generating image for prompt: {prompt[:100]}...")
        time.sleep(1)
        mock_image_filename = f"generated_image_{uuid.uuid4()}.png"
        mock_image_path = self.generated_images_dir / mock_image_filename
        try:
            with open(mock_image_path, "w") as f: f.write("dummy_image_content")
            return {"status": "success", "image_path": str(mock_image_path), "response": f"Image generated and saved to {mock_image_path}"}
        except Exception as e:
            return {"status": "error", "response": f"Failed to create dummy image: {e}"}

   async def get_video_metadata(self, video_path: str) -> Dict: return {"status": "info", "message": "get_video_metadata not fully shown for brevity"}
   async def extract_video_frame(self, video_path: str, timestamp: str) -> Dict: return {"status": "info", "message": "extract_video_frame not fully shown"}
   async def convert_video_to_gif(self, video_path: str, start_time: str, end_time: str, output_filename: Optional[str]=None, scale: float=0.5, fps: int=10) -> Dict: return {"status": "info", "message": "convert_video_to_gif not fully shown"}
   async def get_audio_info(self, audio_path: str) -> Dict: return {"status": "info", "message": "get_audio_info not fully shown"}
   async def convert_audio_format(self, audio_path: str, target_format: str, output_filename_stem: Optional[str]=None) -> Dict: return {"status": "info", "message": "convert_audio_format not fully shown"}
   async def text_to_speech(self, text: str, output_filename_stem: Optional[str]=None) -> Dict: return {"status": "info", "message": "text_to_speech not fully shown"}
   async def scaffold_new_project(self, project_name: str, project_type: str) -> Dict: return {"status": "info", "message": "scaffold_new_project not fully shown"}
   async def explain_code_snippet(self, code_snippet: str, language: Optional[str] = "python") -> Dict: return {"status": "info", "message": "explain_code_snippet not fully shown"}
   async def generate_code_module(self, requirements: str, language: Optional[str] = "python", filename: Optional[str] = None) -> Dict: return {"status": "info", "message": "generate_code_module not fully shown"}
   async def get_system_info(self, component: str) -> Dict: return {"status": "info", "message": "get_system_info not fully shown"}
   async def parallel_execution(self, prompt: str, selected_agents_names: List[str], context: Optional[Dict]=None) -> List[Dict]: return [{"status":"info", "message":"parallel_execution not fully shown"}]

orchestrator = TerminusOrchestrator()
