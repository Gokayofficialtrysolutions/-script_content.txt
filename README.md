# Terminus AI - Ultimate Local AI Ecosystem

Terminus AI is a comprehensive, locally runnable AI ecosystem designed for power users and developers. It provides a suite of tools, numerous AI models, and a flexible agent orchestration system, all intended to run on your local machine.

## Key Features
*   **Local First:** Designed to run primarily on your local hardware.
*   **Ollama Integration:** Seamlessly manages Ollama for running various open-source Large Language Models.
*   **Multiple Pre-configured AI Agents:** Includes specialized agents for tasks like advanced reasoning, code generation, data analysis, web research, and more.
*   **Extensible Agent Configuration:** Agent capabilities and models can be configured via `agents.json`.
*   **Model Management:** Model downloads are managed via `models.conf`, allowing selection of specific models.
*   **Web UI:** Provides an interactive web interface using Streamlit for easy interaction with agents and tools.
*   **Core Capabilities:** Offers document processing, web intelligence, and code generation utilities.
*   **Version-Pinned Dependencies:** Uses `*_requirements.txt` files with pinned versions for Python packages to ensure stability and reproducibility.
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

### Launching Terminus AI
To launch the Terminus AI ecosystem:
```bash
cd ~/.terminus-ai
python3 launch_terminus.py
```
This will start the Ollama server (if not already running) and the Streamlit Web UI. You can access the UI at `http://localhost:8501`.

### Basic Usage
The Web UI provides several operation modes:
*   **Multi-Agent Chat:** Interact with multiple AI agents simultaneously.
*   **Document Processing:** Upload and analyze documents.
*   **Web Intelligence:** Perform web searches and analyze results.
*   Other modes for code generation, data analysis, etc., may be available.

Select agents, adjust parameters like temperature, and input your queries or tasks through the UI.

## Configuration

*   **Models (`models.conf`):** Located in `~/.terminus-ai/models.conf`. This file defines which Ollama models are available for download and allows you to specify core models or a full set.
*   **Agents (`agents.json`):** Located in `~/.terminus-ai/agents.json`. This file configures the available AI agents, their models, specialties, and active status. You can customize this file to add new agents or modify existing ones.

## Note on Dependencies
The project uses version-pinned Python dependencies listed in `core_requirements.txt`, `frameworks_requirements.txt`, `utils_requirements.txt`, and `dev_requirements.txt` to ensure stability. Some newly added dependencies for specialized engines in these files (particularly in `core_requirements.txt` and `utils_requirements.txt` from recent updates) have placeholder versions (e.g., `X.Y.Z`) and will need to be updated to specific, stable versions for full functionality of those engines.
