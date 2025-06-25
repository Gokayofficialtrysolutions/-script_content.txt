#!/bin/bash
# OPUS MAGNUM: ULTIMATE TERMINALIS AI ECOSYSTEM
# Size: ~280GB | Models: 25+ | Capabilities: UNLIMITED
set -e

# Determine the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

INSTALL_DIR="$HOME/.terminus-ai"
LOG="$INSTALL_DIR/install.log"
TOTAL=18 # Adjusted for two progress calls in create_agent_orchestration_script
STEP=0

# Function to display progress
progress(){
    STEP=$((STEP+1))
    echo "[$STEP/$TOTAL-$((STEP*100/TOTAL))%] $1" | tee -a "$LOG"
}

# Function for initial directory creation and logging setup
initialize_setup() {
    progress "INITIALIZING SETUP"
    mkdir -p "$INSTALL_DIR"/{core,models,agents,tools,data,logs,cache}
    touch "$LOG"
    echo "TERMINUS AI: THE ULTIMATE LOCAL AI ECOSYSTEM" | tee -a "$LOG"
    echo "Total: ~280GB | Models: 25+ | Agents: Unlimited" | tee -a "$LOG"

    # Copy models.conf from script directory to INSTALL_DIR
    if [ -f "$SCRIPT_DIR/models.conf" ]; then
        cp "$SCRIPT_DIR/models.conf" "$INSTALL_DIR/models.conf"
        echo "Copied models.conf to $INSTALL_DIR" | tee -a "$LOG"
    else
        echo "WARNING: models.conf not found in script directory ($SCRIPT_DIR). Model configuration will rely on fallback or potentially fail if critical." | tee -a "$LOG"
        # The pull_ollama_models function has its own check for $INSTALL_DIR/models.conf and fallback.
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    progress "INSTALLING SYSTEM DEPENDENCIES" # Corresponds to old "INITIALIZING QUANTUM CORE SYSTEMS" (second part)
    if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install -y python3 python3-pip docker.io git curl wget build-essential cmake ninja-build nodejs npm golang rust-all-dev espeak libespeak1
    fi
    if command -v brew &>/dev/null; then
        brew install python docker git curl wget cmake ninja nodejs go rust espeak
    fi
    # Add more error checking here in later plan steps
}

# No longer defining MODELS array globally. It's handled in pull_ollama_models function.

# Function to install base Python packages like torch, transformers, etc.
install_python_core_libraries() {
    progress "INSTALLING PYTHON CORE LIBRARIES"
    python3 -m pip install --upgrade pip setuptools wheel
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to upgrade pip, setuptools, or wheel. Check $LOG for details." | tee -a "$LOG"
    fi
    echo "Installing Python core libraries from $SCRIPT_DIR/core_requirements.txt" | tee -a "$LOG"
    pip3 install -r "$SCRIPT_DIR/core_requirements.txt" || { echo "ERROR: Failed to install critical packages from $SCRIPT_DIR/core_requirements.txt. Aborting. Check $LOG for details." | tee -a "$LOG"; exit 1; }
}

# Function for Langchain, Autogen, UI frameworks, etc.
install_python_framework_libraries() {
    progress "INSTALLING PYTHON FRAMEWORK LIBRARIES"
    echo "Installing Python framework libraries from $SCRIPT_DIR/frameworks_requirements.txt" | tee -a "$LOG"
    pip3 install -r "$SCRIPT_DIR/frameworks_requirements.txt" || { echo "ERROR: Failed to install critical packages from $SCRIPT_DIR/frameworks_requirements.txt. Aborting. Check $LOG for details." | tee -a "$LOG"; exit 1; }
}

# Function for web scraping, data handling, file processing, etc.
install_python_utility_libraries() {
    progress "INSTALLING PYTHON UTILITY LIBRARIES"
    echo "Installing Python utility libraries from $SCRIPT_DIR/utils_requirements.txt" | tee -a "$LOG"
    pip3 install -r "$SCRIPT_DIR/utils_requirements.txt" || { echo "ERROR: Failed to install critical packages from $SCRIPT_DIR/utils_requirements.txt. Aborting. Check $LOG for details." | tee -a "$LOG"; exit 1; }
}

# Function for installing Ollama
install_ollama_and_dependencies() {
    progress "INSTALLING OLLAMA AND DEPENDENCIES"
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama serve &
    echo "Waiting for Ollama server to start..." | tee -a "$LOG"
    sleep 10 # Increased sleep time for robustness
    if ! ollama list > /dev/null 2>&1 && ! curl -sf --head http://localhost:11434 | grep "HTTP/[12]\.[01] [2].." > /dev/null; then
        echo "ERROR: Ollama server failed to start or is not responding. Aborting. Check $LOG for details." | tee -a "$LOG"
        exit 1
    else
        echo "Ollama server started successfully." | tee -a "$LOG"
    fi
}

# Function for downloading AI models
pull_ollama_models() {
    progress "SELECTING AND PULLING OLLAMA MODELS"

    ALL_AVAILABLE_MODELS=()
    CORE_MODELS=()
    CURRENT_SECTION=""
    CONFIG_FILE="$INSTALL_DIR/models.conf" # Assuming models.conf is in INSTALL_DIR

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Configuration file '$CONFIG_FILE' not found." | tee -a "$LOG"
        echo "Please ensure 'models.conf' exists in $INSTALL_DIR." | tee -a "$LOG"
        echo "Proceeding with no models available for selection. You can only skip model download." | tee -a "$LOG"
        # Fallback: Define minimal core models if config is missing, to prevent errors later if user tries to select core
        CORE_MODELS=("llama3.1:8b" "mistral:7b") # Minimal fallback
    else
        echo "Reading model lists from $CONFIG_FILE..." | tee -a "$LOG"
        while IFS= read -r line || [ -n "$line" ]; do
            # Remove leading/trailing whitespace (optional, but good for robustness)
            line=$(echo "$line" | awk '{$1=$1};1')

            # Skip empty lines and comments
            [[ "$line" =~ ^\s*# ]] && continue
            [[ "$line" =~ ^\s*$ ]] && continue

            if [[ "$line" =~ ^\[(.*)\]$ ]]; then
                CURRENT_SECTION="${BASH_REMATCH[1]}"
            else
                # Remove potential carriage returns for cross-platform compatibility
                line=$(echo "$line" | tr -d '\r')
                if [ -n "$line" ]; then # Ensure line is not empty after stripping CR
                    case "$CURRENT_SECTION" in
                        ALL_AVAILABLE_MODELS)
                            ALL_AVAILABLE_MODELS+=("$line")
                            ;;
                        CORE_MODELS)
                            CORE_MODELS+=("$line")
                            ;;
                    esac
                fi
            fi
        done < "$CONFIG_FILE"
        echo "Finished reading model lists. Found ${#ALL_AVAILABLE_MODELS[@]} available models and ${#CORE_MODELS[@]} core models." | tee -a "$LOG"
    fi

    if [ ${#ALL_AVAILABLE_MODELS[@]} -eq 0 ] && [ -f "$CONFIG_FILE" ]; then
        echo "WARNING: No models were loaded from $CONFIG_FILE. It might be empty or incorrectly formatted." | tee -a "$LOG"
        echo "Model download options will be limited. You may only be able to skip." | tee -a "$LOG"
    elif [ ${#ALL_AVAILABLE_MODELS[@]} -eq 0 ] && [ ! -f "$CONFIG_FILE" ]; then
        # Error already printed, this is just to ensure the flow is logical
        echo "Continuing with no models defined due to missing models.conf." | tee -a "$LOG"
    fi

    # Ensure CORE_MODELS is not empty if user might select it, even if ALL_AVAILABLE_MODELS is empty.
    # This is a safety net, though the user prompt should guide them.
    if [ ${#CORE_MODELS[@]} -eq 0 ] && [ ${#ALL_AVAILABLE_MODELS[@]} -gt 0 ]; then
        echo "WARNING: CORE_MODELS list is empty in models.conf. Selecting 'CORE' will result in no models being downloaded unless ALL models are also empty." | tee -a "$LOG"
    elif [ ${#CORE_MODELS[@]} -eq 0 ] && [ ${#ALL_AVAILABLE_MODELS[@]} -eq 0 ]; then
         # If both are empty (e.g. models.conf missing and no fallback for CORE_MODELS or ALL_AVAILABLE_MODELS)
         CORE_MODELS=("llama3.1:8b" "mistral:7b") # Re-apply minimal fallback for safety if somehow cleared
         echo "Re-applying minimal fallback for CORE_MODELS as both lists were empty." | tee -a "$LOG"
    fi


    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    echo "Ollama Model Installation Options:" | tee -a "$LOG"
    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    echo "Available models for installation:" | tee -a "$LOG"
    for i in "${!ALL_AVAILABLE_MODELS[@]}"; do
        printf "  %2d. %s\n" "$((i+1))" "${ALL_AVAILABLE_MODELS[$i]}" | tee -a "$LOG"
    done
    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    echo "You can choose to:" | tee -a "$LOG"
    echo "  1. Download ALL available models (${#ALL_AVAILABLE_MODELS[@]} models, ~180GB+)." | tee -a "$LOG"
    echo "  2. Download a CORE set of essential models (${#CORE_MODELS[@]} models, ~20-50GB)." | tee -a "$LOG"
    echo "  3. Select specific models to download." | tee -a "$LOG"
    echo "  4. Skip Ollama model downloads for now." | tee -a "$LOG"
    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    read -r -p "Enter your choice (1, 2, 3, or 4): " user_choice

    MODELS_TO_PULL=()
    case "$user_choice" in
        1)
            echo "Preparing to download ALL ${#ALL_AVAILABLE_MODELS[@]} models." | tee -a "$LOG"
            MODELS_TO_PULL=("${ALL_AVAILABLE_MODELS[@]}")
            ;;
        2)
            echo "Preparing to download CORE set of ${#CORE_MODELS[@]} models." | tee -a "$LOG"
            MODELS_TO_PULL=("${CORE_MODELS[@]}")
            ;;
        3)
            echo "Enter the names of the models you wish to download, separated by spaces." | tee -a "$LOG"
            echo "Example: llama3.1:8b mistral:7b deepseek-coder-v2:16b" | tee -a "$LOG"
            echo "Available models listed above. Please type or copy-paste exact names." | tee -a "$LOG"
            read -r -p "Selected models: " selected_models_str
            # Convert the space-separated string to an array
            read -r -a USER_SELECTED_MODELS <<< "$selected_models_str"

            # Validate user selections
            for model_name in "${USER_SELECTED_MODELS[@]}"; do
                is_valid=false
                for available_model in "${ALL_AVAILABLE_MODELS[@]}"; do
                    if [[ "$model_name" == "$available_model" ]]; then
                        MODELS_TO_PULL+=("$model_name")
                        is_valid=true
                        break
                    fi
                done
                if ! $is_valid; then
                    echo "WARNING: Model '$model_name' is not in the list of available models and will be skipped." | tee -a "$LOG"
                fi
            done

            if [ ${#MODELS_TO_PULL[@]} -eq 0 ] && [ ${#USER_SELECTED_MODELS[@]} -ne 0 ]; then
                 echo "No valid models selected from your input. Defaulting to CORE models." | tee -a "$LOG"
                 MODELS_TO_PULL=("${CORE_MODELS[@]}")
            elif [ ${#MODELS_TO_PULL[@]} -eq 0 ]; then
                 echo "No models selected. Defaulting to CORE models." | tee -a "$LOG"
                 MODELS_TO_PULL=("${CORE_MODELS[@]}")
            fi
            ;;
        4)
            echo "Skipping Ollama model downloads as per user choice." | tee -a "$LOG"
            # MODELS_TO_PULL will remain empty
            ;;
        *)
            echo "Invalid choice. Defaulting to CORE set of models." | tee -a "$LOG"
            MODELS_TO_PULL=("${CORE_MODELS[@]}")
            ;;
    esac

    if [ ${#MODELS_TO_PULL[@]} -eq 0 ]; then
        echo "No models selected for download. Skipping Ollama model pulling phase." | tee -a "$LOG"
        return
    fi

    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    echo "The following ${#MODELS_TO_PULL[@]} models will be downloaded:" | tee -a "$LOG"
    for model_to_pull in "${MODELS_TO_PULL[@]}"; do
        echo "- $model_to_pull" | tee -a "$LOG"
    done
    echo "----------------------------------------------------------------------" | tee -a "$LOG"
    sleep 3 # Give user time to read

    FAILED_MODELS=()
    SUCCESSFUL_MODELS=0
    TOTAL_MODELS_TO_PULL=${#MODELS_TO_PULL[@]}

    for model in "${MODELS_TO_PULL[@]}";do # Iterate over MODELS_TO_PULL
        echo "Pulling $model ($((SUCCESSFUL_MODELS + ${#FAILED_MODELS[@]} + 1))/$TOTAL_MODELS_TO_PULL)..." | tee -a "$LOG"
        ollama pull "$model"
        if [ $? -ne 0 ]; then
            echo "WARNING: Failed to pull model $model. It will be skipped. Check $LOG for details." | tee -a "$LOG"
            FAILED_MODELS+=("$model")
        else
            echo "Successfully pulled $model." | tee -a "$LOG"
            SUCCESSFUL_MODELS=$((SUCCESSFUL_MODELS + 1))
        fi
    done

    echo "Ollama model pulling complete. $SUCCESSFUL_MODELS/$TOTAL_MODELS models downloaded successfully." | tee -a "$LOG"

    if [ ${#FAILED_MODELS[@]} -ne 0 ]; then
        echo "Summary of failed model downloads (${#FAILED_MODELS[@]}):" | tee -a "$LOG"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "- $failed_model" | tee -a "$LOG"
        done
    fi
}

# Function for generating master_orchestrator.py
create_agent_orchestration_script() {
    progress "CREATING AGENT CONFIGURATION FILE (agents.json)"
    cat>"$INSTALL_DIR/agents.json"<<'AGENTS_EOF'
[
  {
    "name": "DeepThink",
    "model": "deepseek-r1:32b",
    "specialty": "Advanced Reasoning & Logic",
    "active": true
  },
  {
    "name": "MasterPlanner",
    "model": "mixtral:8x22b",
    "specialty": "Complex Task Planning and Decomposition into Agent Steps. Output ONLY JSON plans.",
    "active": true
  },
  {
    "name": "CodeMaster",
    "model": "deepseek-coder-v2:16b",
    "specialty": "Programming & Development",
    "active": true
  },
  {
    "name": "DataWizard",
    "model": "qwen2.5:72b",
    "specialty": "Data Analysis & Processing",
    "active": true
  },
  {
    "name": "WebCrawler",
    "model": "dolphin-mixtral:8x7b",
    "specialty": "Web Research & Intelligence",
    "active": true
  },
  {
    "name": "DocProcessor",
    "model": "llama3.1:70b",
    "specialty": "Document Analysis & Generation",
    "active": true
  },
  {
    "name": "VisionAI",
    "model": "llava:34b",
    "specialty": "Image & Visual Processing",
    "active": true
  },
  {
    "name": "MathGenius",
    "model": "deepseek-math:7b",
    "specialty": "Mathematical Computations",
    "active": true
  },
  {
    "name": "CreativeWriter",
    "model": "nous-hermes2:34b",
    "specialty": "Creative Content Generation",
    "active": true
  },
  {
    "name": "SystemAdmin",
    "model": "codellama:34b",
    "specialty": "System Administration",
    "active": true
  },
  {
    "name": "SecurityExpert",
    "model": "mixtral:8x22b",
    "specialty": "Cybersecurity Analysis",
    "active": true
  },
  {
    "name": "ResearchBot",
    "model": "yi:34b",
    "specialty": "Scientific Research",
    "active": true
  },
  {
    "name": "MultiLang",
    "model": "qwen2.5-coder:32b",
    "specialty": "Multilingual Processing",
    "active": true
  },
  {
    "name": "ImageForge",
    "model": "diffusers/stable-diffusion-xl-base-1.0",
    "specialty": "Image Generation",
    "active": true
  },
  {
    "name": "AudioMaestro",
    "model": "pydub/pyttsx3",
    "specialty": "Audio Processing & TTS",
    "active": true
  },
  {
    "name": "ContentAnalysisAgent",
    "model": "llama3.1:70b",
    "specialty": "Performs deeper analysis of text content (e.g., keyword extraction, topic modeling) to enrich knowledge base entries. Often triggered by system events.",
    "active": true
  }
]
AGENTS_EOF
    echo "Created agents.json in $INSTALL_DIR" | tee -a "$LOG"

    progress "DEPLOYING AGENT ORCHESTRATION SCRIPT (master_orchestrator.py)"
    # Ensure the target directory exists
    mkdir -p "$INSTALL_DIR/agents"
    if [ -f "$SCRIPT_DIR/src/agents/master_orchestrator.py" ]; then
        cp "$SCRIPT_DIR/src/agents/master_orchestrator.py" "$INSTALL_DIR/agents/master_orchestrator.py"
        echo "Copied src/agents/master_orchestrator.py to $INSTALL_DIR/agents/" | tee -a "$LOG"
    else
        echo "ERROR: Source file src/agents/master_orchestrator.py not found in $SCRIPT_DIR. Orchestrator script cannot be deployed." | tee -a "$LOG"
        # Decide if this is a fatal error
        # exit 1
    fi
}

# Function for generating terminus_ui.py
create_terminus_ui_script() {
    progress "DEPLOYING TERMINUS UI SCRIPT"
    if [ -f "$SCRIPT_DIR/src/terminus_ui.py" ]; then
        cp "$SCRIPT_DIR/src/terminus_ui.py" "$INSTALL_DIR/terminus_ui.py"
        echo "Copied src/terminus_ui.py to $INSTALL_DIR/" | tee -a "$LOG"
    else
        echo "ERROR: Source file src/terminus_ui.py not found in $SCRIPT_DIR. UI script cannot be deployed." | tee -a "$LOG"
        # Decide if this is a fatal error
        # exit 1
    fi
}
