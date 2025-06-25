# Terminalis AGI - Ultimate Local AI Ecosystem

Terminalis AGI is a comprehensive, locally runnable AI ecosystem designed for power users and developers. It provides a suite of tools, numerous AI models, and a flexible agent orchestration system, all intended to run on your local machine.

## Key Features
*   **Local First:** Designed to run primarily on your local hardware.
*   **Ollama Integration:** Seamlessly manages Ollama for running various open-source Large Language Models.
*   **Multiple Pre-configured AI Agents:** Includes specialized agents for tasks like advanced reasoning, code generation, data analysis, web research, and more.
    *   **ImageForge Agent:** Text-to-image generation using advanced diffusion models.
    *   **Video Processing Utilities:** Get video info, extract frames, and convert video segments to GIF.
    *   **AudioMaestro Agent:** Audio file analysis (get info), format conversion, and Text-to-Speech (TTS).
    *   **Enhanced CodeMaster:** Experimental AI-assisted code modification in projects, code explanation, and generation of new code modules/classes.
*   **Sophisticated Orchestration Engine:**
    *   **MasterPlanner Agent (Evolving):** Decomposes complex user requests into multi-step execution plans. Features include:
        *   Step-level validation and robust retry mechanisms.
        *   Basic parallel execution for independent tasks.
        *   Targeted plan revision based on detailed failure context.
        *   Preliminary Knowledge Base query to inform planning.
    *   **Contextual Agent Selection:** Orchestrator utilizes UI context and zero-shot intent classification for improved agent routing.
    *   **Conversational Memory:** Basic conversation history is maintained and provided as context to MasterPlanner.
*   **Persistent Knowledge Base (ChromaDB):**
    *   Stores and retrieves textual information (document excerpts, web summaries, code explanations/generations) via semantic search.
    *   Includes a "Knowledge Base Explorer" in the Web UI for direct user querying.
    *   Automatically augmented by agents like DocumentProcessor (via UI), WebCrawler, and CodeMaster.
*   **Extensible Agent Configuration:** Agent capabilities and models can be configured via `agents.json`.
*   **Model Management:** Model downloads are managed via `models.conf`.
*   **Web UI (Streamlit):** Provides an interactive interface with various operation modes, including the new KB Explorer.
*   **Core Capabilities:** Document processing, web intelligence, project scaffolding, system information.
*   **Version-Pinned Dependencies:** Uses `*_requirements.txt` files with pinned versions for Python packages to ensure stability and reproducibility. All dependencies are pinned to specific versions.
*   **Automated Setup:** A single bash script automates the installation of dependencies and components.

## Installation

### System Requirements
*   A Linux or macOS environment.
*   Python 3.8+ and Pip.
*   Docker (for some potential future integrations, though primarily uses local Ollama).
*   Standard build tools (git, curl, wget, build-essential, cmake, etc. - the installer attempts to manage these).
*   Sufficient disk space (min. 100GB recommended for multiple models).

### Running the Installer
1.  Ensure you have the installer script (e.g., `script_content.txt` or a named `.sh` file).
2.  Make it executable: `chmod +x your_script_name.sh`
3.  Run the installer: `bash your_script_name.sh`

The main installation directory will be `~/.terminus-ai`. All logs and generated files will be stored there.

## Post-Installation

### Launching Terminalis AGI
To launch the Terminalis AGI ecosystem:
```bash
cd ~/.terminus-ai
python3 launch_terminus.py
```
This will start the Ollama server (if not already running) and the Streamlit Web UI. You can access the UI at `http://localhost:8501`.

### Basic Usage
The Web UI provides several operation modes:
*   **Multi-Agent Chat:** Interact with multiple AI agents simultaneously. Can optionally use `MasterPlanner` for complex requests and benefits from contextual agent selection and conversational memory.
*   **Document Processing:** Upload and analyze documents.
*   **Web Intelligence:** Perform web searches and analyze results.
*   **Image Generation:** Create images from text prompts.
*   **Video Processing:** Utilities for video file information, frame extraction, and GIF conversion.
*   **Audio Processing:** Tools for audio analysis, format conversion, and Text-to-Speech.
*   **Code Generation:** Scaffold new projects, modify existing code (experimental), explain code snippets, and generate code modules. (Explanations & generated modules are now stored in the Knowledge Base).
*   **Knowledge Base Explorer:** Directly query and explore the contents of the system's persistent knowledge base.
*   Other modes for data analysis, system information, etc., may be available.

Select agents, adjust parameters like temperature, and input your queries or tasks through the UI.

## Configuration

*   **Models (`models.conf`):** Located in `~/.terminus-ai/models.conf`. This file defines which Ollama models are available for download and allows you to specify core models or a full set.
*   **Agents (`agents.json`):**
    *   **Production:** Located in `~/.terminus-ai/agents.json` after installation. This file configures the available AI agents, their models, specialties, and active status. You can customize this file to add new agents or modify existing ones.
    *   **Development:** A default `src/agents.json` is provided in the source repository. This allows the application to run with a predefined set of agents when executed directly from the source tree (e.g., for development or testing). The installer script is responsible for handling the `agents.json` file in the production installation directory (`~/.terminus-ai`).

## Note on Dependencies
The project uses version-pinned Python dependencies listed in `core_requirements.txt`, `frameworks_requirements.txt`, `utils_requirements.txt`, and `dev_requirements.txt`. All dependencies are pinned to specific, stable versions to ensure stability and reproducibility of the environment.
