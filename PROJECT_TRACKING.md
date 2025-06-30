# Terminalis AGI - Project Tracking

## Section 1: "Terminalis AGI - Version 1 Status (Current)"

**Key Features (Functionally Complete for V1):**
*   [X] Ollama Integration
*   [X] ImageForge Agent (Text-to-image generation)
*   [X] Video Processing Utilities (Get info, extract frames, convert to GIF)
*   [X] AudioMaestro Agent (Audio analysis, format conversion, TTS)
*   [X] Enhanced CodeMaster (AI-assisted code modification, explanation, module generation)
*   [X] MasterPlanner Agent (Experimental - multi-step plan execution)
*   [X] Contextual Agent Selection (UI context & zero-shot intent classification)
*   [X] Conversational Memory (Basic history for MasterPlanner)
*   [X] Web UI (Streamlit based)
*   [X] Core Capabilities (Document processing, web intelligence, project scaffolding)
*   [X] Automated Setup Script - *Core Python components (Orchestrator, UI) now managed as separate source files and copied by installer, improving maintainability.*
*   [X] Version-Pinned Dependencies (via `*_requirements.txt`)
*   [X] Externalized Configuration (`agents.json`, `models.conf`)

**Version 1 Polish Items:**
*   [X] Resolve `passlib` dependency version (Completed in Plan Step 1: `passlib==1.7.4`)
*   [ ] Comprehensive integrated testing (Ongoing user responsibility / future dedicated effort)

**Overall V1 Progress:** 100% (Core features implemented and polish items completed. Comprehensive integrated testing remains an ongoing user/team responsibility).

**Key V1.x Enhancements / Foundational Roadmap Progress (Post V1 Core):**
*   [X] MasterPlanner: Implemented step-level plan validation.
*   [X] MasterPlanner: Implemented robust step-level retry logic (max_retries, delay, retry_on_statuses).
*   [X] Knowledge Base: Core backend implemented (ChromaDB for persistent vector storage, store/retrieve methods in Orchestrator).
*   [X] Knowledge Base: UI for direct user exploration and querying of KB content.
*   [X] MasterPlanner: Basic parallel execution capability for independent plan steps.
*   [X] MasterPlanner: Enhanced plan revision process with targeted failure context provided to LLM.
*   [X] Knowledge Base: Automated augmentation by WebCrawler (stores summaries of scraped pages).
*   [X] Knowledge Base: Automated augmentation by CodeMaster (stores code explanations and generated modules).
*   [X] Knowledge Base: `MasterPlanner` now performs a preliminary KB query to fetch relevant context before generating a plan.
*   [X] MasterPlanner: Leverages extracted keywords from KB items (when available) to improve contextual understanding during planning.
*   [X] Inter-Agent Communication: Basic asynchronous message bus implemented.
*   [X] Knowledge Base: Automated content analysis (keyword extraction) for new KB items via message bus.
*   [X] Knowledge Base: Automated content analysis now includes topic modeling; MasterPlanner utilizes topics.
*   [X] User Feedback: System can analyze collected feedback and store a summary report in the Knowledge Base.
*   [X] MasterPlanner: Utilizes feedback analysis reports from KB to adapt planning.
*   [X] System Integration: Performed conceptual integrated testing of major components (MasterPlanner, KB, Message Bus, Feedback).
*   [X] System Robustness: Implemented improved error logging and validation for key orchestrator methods.

## Section 2: "Roadmap: Towards an Ultimate AGI Orchestrator"

**Overall AGI Orchestrator Progress:** ~28-33% (Includes KG storage/retrieval of simplified plans for MasterPlanner context, and expanded core testing).

### Phase X: Advanced Cognitive Orchestration
*Key Objectives:*
*   [X] **Enhanced `MasterPlanner` Capabilities (Evolution):** (Initial capabilities implemented, learning mechanisms enhanced)
    *   [X] Complex Plan Execution (parallel, conditional, looping/retry) - *Initial retry and parallel execution implemented.*
    *   [P] Dynamic Plan Adaptation & Learning - *Initial targeted revision context and learning from feedback reports implemented. Topic utilization from KB, and retrieval of simplified past plan structures from KG also enhance adaptation and learning.*
    *   [P] Resource-Aware & Prioritized Planning - *Agent `estimated_speed` added; MasterPlanner prompt updated for speed, complexity, and request priority.*
*   [X] **Sophisticated Intent Understanding & Contextual Awareness:** (Initial NLU enhancements implemented)
    *   [X] Nuanced NLU (ambiguity, implicit goals, long-term context) - *Enhanced with NER, alternative intent suggestion, implicit goal extraction; MasterPlanner KB pre-query leverages this. MasterPlanner prompt updated for richer NLU.*
    *   [ ] Dedicated NLU module/agent
    *   [ ] Richer contextual model of interaction & user objectives

### Phase Y: System Learning, Memory & Adaptation
*Key Objectives:*
*   [P] **Persistent & Adaptive Knowledge Base:** (Core ChromaDB implemented, Graph DB for plan summaries and relationships)
    *   [X] Store, retrieve, manage diverse knowledge (facts, procedures, preferences, learned associations) - *ChromaDB backend, store/retrieve methods, UI explorer implemented. Data types include doc excerpts, web summaries, code explanations/generations, plan execution logs, feedback reports.*
    *   [X] Automated content enrichment (keyword & topic extraction) for most new KB entries.
    *   [P] Hybrid DB approach (vector + graph) - *SQLite-based graph DB stores relationships (item-topic/keyword, plan-feedback) and simplified plan structures. MasterPlanner retrieves related items and past simplified plans from KG to inform planning.*
*   [X] **Comprehensive User Feedback Loop & Reinforcement Learning:** (Feedback collection, analysis, and planner utilization implemented)
    *   [X] Explicit user feedback mechanisms (ratings, corrections) - *UI widgets for feedback integrated across relevant UI outputs.*
    *   [X] System analyzes feedback to generate summary reports, stored in KB.
    *   [X] MasterPlanner actively uses feedback reports from KB for plan adaptation.
    *   [ ] RL frameworks for adjusting behavior, prompts, tool selection - *Initial design for simple value-averaging RL based on existing logger completed. No implementation yet.*
*   [X] **Knowledge Ingestion & Synthesis:** (Agent-driven ingestion, reactive analysis, planner utilization)
    *   [X] Autonomous ingestion from documents/links - *WebCrawler, DocumentProcessor, CodeMaster store outputs in KB.*
    *   [X] Reactive analysis of new KB content (keyword extraction, topic modeling) triggered via message bus.
    *   [X] MasterPlanner utilizes extracted keywords and topics from KB items for contextual planning.
    *   [ ] Summarization & indexing of ingested data - *Basic summarization for WebCrawler; indexing inherent to ChromaDB. Keyword/topic extraction enhances discoverability.*

### Phase Z: Expanded Autonomy & Collaborative Intelligence
*Key Objectives:*
*   [P] **Advanced Tool Discovery & Augmentation:**
    *   [ ] Dynamically discover, evaluate, learn new tools/APIs
    *   [P] Agents request/suggest new tool integrations - *MasterPlanner can now suggest new tools via a `SystemCapabilityManager` step; suggestions are logged to `logs/tool_suggestions.log`.*
*   [X] **Sophisticated Inter-Agent Communication & Collaboration:** (Initial foundation laid & first use case verified)
    *   [X] Basic asynchronous message bus implemented, enabling event broadcasting and subscription by agents/components.
        *   *Initial use case: KB event triggers `ContentAnalysisAgent` for keyword extraction from new KB entries.*
        *   *Verified `agent_service_call` for task delegation (e.g., `DocSummarizer.summarize_text` service).*
    *   [ ] Advanced protocols for negotiation, complex info sharing, coordination
    *   [ ] Multi-agent consensus, task delegation, shared goal understanding
*   [ ] **Proactive & Goal-Oriented Assistance:**
    *   [ ] Proactive suggestions, workflow optimization, need anticipation
    *   [ ] Autonomous work towards user-defined high-level objectives

### Phase W: System Self-Understanding & Meta-Cognition (Research Intensive)
*Key Objectives:*
*   [ ] **Environmental & System Awareness (includes `dxdiag`-like info):**
    *   [ ] Dynamic model of user's computing environment (software, hardware, data)
*   [ ] **Dynamic Capability Mapping & Performance Monitoring:**
    *   [ ] Internal map of agent skills, dependencies, reliability/performance
    *   [ ] Track success rates, processing times, resource consumption
*   [ ] **Adaptive Self-Optimization (Long-Term, Experimental):**
    *   [ ] Identify internal bottlenecks
    *   [ ] System suggests/implements optimizations (workflows, agent logic, prompts)
*   [ ] **Dynamic Resource Management & Allocation:**
    *   [ ] Sophisticated strategies for allocating CPU, GPU, memory

### Phase V: Towards Deeper AI Integration & "Neural Connections"
*Key Objectives:*
*   [ ] **Advanced AI Architectures & Neuro-Symbolic Integration:**
    *   [ ] Research/prototype neuro-symbolic AI
    *   [ ] Diverse neural architectures (GNNs, advanced perception)
*   [ ] **Adaptive User Interaction & Dialogue Management:**
    *   [ ] Learn user communication styles, preferences, terminology
    *   [ ] More collaborative, conversational interaction
*   [ ] **Enhanced Creative & Generative Capabilities:**
    *   [ ] Integrate models for video generation, music composition, 3D assets

### Phase U: Foundational AGI Research & Ethical Frameworks
*Key Objectives:*
*   [ ] **AGI Concepts & Cognitive Architectures:**
    *   [ ] Continuous research and integration of AGI concepts
*   [ ] **Formalized Ethical AI & Safety Protocols:**
    *   [ ] Develop/implement safety, value alignment, explainability, controllability
*   [X] **Resilience & Robustness:**
    *   [X] Graceful handling of unforeseen circumstances, novel problems - *Implemented improved error logging and validation for key orchestrator methods (feedback analysis, content analysis).*

### Cross-Cutting Concerns (Continuous Development)
*   [ ] Security & Sandboxing
*   [ ] Ethical Alignment & User Control
*   [ ] Performance Optimization & Efficiency
*   [ ] Dependency Management & Upgradability
*   [P] Comprehensive Testing & Validation - *Test framework setup with Pytest. Unit tests implemented for `KnowledgeGraph`, core `TerminusOrchestrator` logic (`classify_user_intent`, `_evaluate_plan_condition`), ChromaDB interactions (`store_knowledge`, `retrieve_knowledge`), and Event Bus/core handlers (`_event_handler_kb_content_added`).*
*   [ ] Documentation & Community
