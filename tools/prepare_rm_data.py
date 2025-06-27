import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # For saving preprocessor

# Define known categorical features and their potential categories based on design
# This helps ensure consistent encoding, especially if some categories are not in the initial dataset.
# These should align with what TerminusOrchestrator / RLExperienceLogger can produce.
KNOWN_CATEGORICAL_FEATURES = {
    "nlu_intent": [ # From TerminusOrchestrator.candidate_intent_labels
        "image_generation", "code_generation", "code_modification", "code_explanation",
        "project_scaffolding", "video_info", "video_frame_extraction", "video_to_gif",
        "audio_info", "audio_format_conversion", "text_to_speech",
        "data_analysis", "web_search", "document_processing", "general_question_answering",
        "complex_task_planning", "system_information_query", "knowledge_base_query",
        "feedback_submission", "feedback_analysis_request", "agent_service_call", "unknown_intent", "none"
    ],
    "user_prompt_length_category": ["short", "medium", "long"],
    "previous_cycle_outcome": ["success", "failure", "none"], # From RL state design
    "request_priority": ["high", "normal", "low", "unknown"], # From RL state design
    "action_taken": [ # From RL action space design
        "Action_DefaultPrompt", "Action_EmphasizeSimplicity", "Action_EncourageDetail",
        "Action_PrioritizeKBSearch", "Action_FocusOnNLUIntent",
        "Action_DefaultPrompt_InitialLog", # Placeholder used during initial logging
        "Action_Planner_Error_Before_Action" # Placeholder
    ],
    "plan_parsing_status": ["success", "failure", "unknown"], # Added unknown
    "execution_status": ["success", "failure", "partial_success", "not_executed", "unknown"], # Added unknown
    # Boolean features will be converted to 0/1 numeric directly
}

# Numerical features that might need scaling
NUMERICAL_FEATURES = [
    "nlu_entities_count", "kb_general_hits", "kb_plan_log_hits", "kb_feedback_hits",
    "plan_num_steps", "plan_max_depth", "num_revisions_for_plan", "plan_avg_complexity_score"
]

# Boolean features to be converted to int (0 or 1)
BOOLEAN_FEATURES = [
    "plan_uses_conditional", "plan_uses_loop", "plan_uses_parallel", "plan_uses_service_call"
]


def extract_plan_features(plan_json_str: Optional[str]) -> Dict[str, Any]:
    """
    Parses a plan JSON string and extracts features.
    Returns a dictionary of features.
    """
    features = {
        "plan_num_steps": 0,
        "plan_max_depth": 0,
        "plan_uses_conditional": 0,
        "plan_uses_loop": 0,
        "plan_uses_parallel": 0,
        "plan_uses_service_call": 0,
        "plan_avg_complexity_score": 0.0
    }
    if not plan_json_str or plan_json_str == "N/A" or plan_json_str == "[]": # Handle empty plan string
        return features

    try:
        plan_list = json.loads(plan_json_str)
        if not isinstance(plan_list, list) or not plan_list: # Also check if plan_list is empty after parsing
            return features

        features["plan_num_steps"] = len(plan_list)

        max_depth_found = 0
        if features["plan_num_steps"] > 0:
            max_depth_found = 1 # At least one level if steps exist

        for step in plan_list:
            if not isinstance(step, dict): continue
            step_type = step.get("step_type")
            if step_type == "conditional": features["plan_uses_conditional"] = 1
            if step_type == "loop": features["plan_uses_loop"] = 1
            if step_type == "agent_service_call": features["plan_uses_service_call"] = 1
            if step.get("agent_name") == "parallel_group":
                features["plan_uses_parallel"] = 1
                if max_depth_found < 2: max_depth_found = 2

        features["plan_max_depth"] = max_depth_found
        # Note: plan_avg_complexity_score would require access to agent definitions to map agent names to complexity.
        # This is a simplification for now. A more advanced version would load agents.json.

    except json.JSONDecodeError:
        print(f"Warning: Could not parse plan JSON for feature extraction: {plan_json_str[:100]}...")
    except Exception as e:
        print(f"Warning: Error extracting plan features: {e}")

    return features

def load_and_preprocess_data(log_file_path: Path) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    experiences = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                experiences.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line {line_num} in {log_file_path}")
                continue

    if not experiences:
        # Return empty structures if no valid experiences
        return pd.DataFrame(), pd.Series(dtype=float), ColumnTransformer(transformers=[], remainder='drop')


    processed_data = []
    for exp in experiences:
        state = exp.get("state", {})
        action = exp.get("action_taken", "Action_Unknown")
        plan_json_to_analyze = exp.get("final_executed_plan_json") or exp.get("generated_plan_json")
        plan_features = extract_plan_features(plan_json_to_analyze)

        features = {
            "nlu_intent": state.get("nlu_intent", "none"),
            "nlu_entities_count": state.get("nlu_entities_count", 0),
            "user_prompt_length_category": state.get("user_prompt_length_category", "short"),
            "kb_general_hits": state.get("kb_general_hits", 0),
            "kb_plan_log_hits": state.get("kb_plan_log_hits", 0),
            "kb_feedback_hits": state.get("kb_feedback_hits", 0),
            "previous_cycle_outcome": state.get("previous_cycle_outcome", "none"),
            "request_priority": state.get("request_priority", "unknown"),
            "action_taken": action,
            "plan_parsing_status": exp.get("plan_parsing_status", "unknown"),
            "execution_status": exp.get("execution_status", "unknown"),
            "num_revisions_for_plan": exp.get("attempt_number", 0),
            **plan_features
        }

        target_reward = exp.get("calculated_reward")
        if target_reward is None:
            print(f"Warning: Missing 'calculated_reward' in log entry: {exp.get('rl_interaction_id')}")
            continue

        processed_data.append({**features, "target_reward": float(target_reward)})

    if not processed_data:
        return pd.DataFrame(), pd.Series(dtype=float), ColumnTransformer(transformers=[], remainder='drop')

    df = pd.DataFrame(processed_data)

    for bf in BOOLEAN_FEATURES:
        if bf in df.columns: df[bf] = df[bf].astype(int)
        else: df[bf] = 0

    current_numerical_features = [nf for nf in NUMERICAL_FEATURES if nf in df.columns]
    current_categorical_features = []
    for cf_name, known_cats in KNOWN_CATEGORICAL_FEATURES.items():
        if cf_name not in df.columns: df[cf_name] = "unknown" # Add column if missing
        # Ensure all known categories are present for OneHotEncoder, even if not in current data subset
        df[cf_name] = pd.Categorical(df[cf_name], categories=known_cats)
        current_categorical_features.append(cf_name)

    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # Ensure correct categories are passed if using pd.Categorical
    # For OneHotEncoder, 'categories' param should be a list of lists of categories
    ohe_categories = [df[col].cat.categories.tolist() for col in current_categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, current_numerical_features),
            ('cat', OneHotEncoder(categories=ohe_categories, handle_unknown='ignore', sparse_output=False), current_categorical_features)
        ],
        remainder='drop' # Drop columns not specified (like original plan_json etc.)
    )

    X = df.drop("target_reward", axis=1)
    y = df["target_reward"]

    return X, y, preprocessor


def main(log_file: str, output_dir: str, preprocessor_path: str):
    log_file_path = Path(log_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessor_file = Path(preprocessor_path)

    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        print(f"Log file {log_file_path} is empty or does not exist. No data to process.")
        # Create empty files to satisfy downstream dependencies if needed, or handle appropriately
        # For now, just exit.
        return

    print(f"Loading data from: {log_file_path}")
    X, y, preprocessor = load_and_preprocess_data(log_file_path)

    if X.empty:
        print("No processable data found after loading. Exiting.")
        return

    print(f"Processed {len(X)} samples.")

    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X)
    print(f"Processed feature shape: {X_processed.shape}")

    joblib.dump(preprocessor, preprocessor_file)
    print(f"Preprocessor saved to {preprocessor_file}")

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y.values, test_size=0.2, random_state=42, stratify=None if len(y.unique()) <=1 else y) # Stratify if classification
    print(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val.npy", y_val)

    try:
        feature_names = preprocessor.get_feature_names_out()
        with open(output_path / "feature_names.json", 'w') as f:
            json.dump(feature_names.tolist(), f)
        print(f"Feature names saved to {output_path / 'feature_names.json'}")
    except Exception as e:
        print(f"Could not save feature names (likely due to empty data or preprocessor issue): {e}")

    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RL experience data for Reward Model training.")
    parser.add_argument("--log_file", type=str, default="../logs/rl_experience_log.jsonl",
                        help="Path to the rl_experience_log.jsonl file.")
    parser.add_argument("--output_dir", type=str, default="../data/rm_training_data",
                        help="Directory to save processed training/validation data.")
    parser.add_argument("--preprocessor_path", type=str, default="../data/rm_training_data/rm_preprocessor.joblib",
                        help="Path to save the fitted scikit-learn preprocessor.")

    args = parser.parse_args()
    main(args.log_file, args.output_dir, args.preprocessor_path)
