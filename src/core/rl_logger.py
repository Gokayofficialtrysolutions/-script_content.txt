import json
from pathlib import Path
from typing import Dict, Optional, List, Any # Added Any
import datetime # For timestamp_end_iso if not passed in explicitly for each log
import uuid # For rl_interaction_id if generated here, though plan is to pass it
import time # For example usage

class RLExperienceLogger:
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        # Ensure the log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experience(self,
                       rl_interaction_id: str,
                       attempt_number: int,
                       state: Dict[str, Any],
                       action: str,
                       master_planner_prompt_details: Dict[str, Any],
                       generated_plan_json: str,
                       plan_parsing_status: str,
                       # Fields below are for the outcome of the interaction cycle / executed attempt
                       final_executed_plan_json: Optional[str],
                       execution_status: str, # e.g., 'success', 'failure', 'partial_success'
                       user_feedback_rating: Optional[str], # e.g., 'positive', 'negative', 'none'
                       calculated_reward: float,
                       next_state: Optional[Dict[str, Any]],
                       done: bool,
                       timestamp_start_iso: str, # When the MasterPlanner cycle began for this S,A
                       timestamp_end_iso: str   # When the cycle (incl. execution & reward calc) ended
                      ) -> None:
        """
        Logs a complete RL experience tuple to a JSONL file.
        Each call appends a new line (a JSON object) to the file.
        """
        experience_log_entry = {
            "rl_interaction_id": rl_interaction_id,
            "attempt_number": attempt_number,
            "timestamp_start_iso": timestamp_start_iso, # Logged when S,A is decided
            "state": state,
            "action_taken": action,
            "master_planner_prompt_details": master_planner_prompt_details,
            "generated_plan_json": generated_plan_json,
            "plan_parsing_status": plan_parsing_status,
            "final_executed_plan_json": final_executed_plan_json, # Plan that was actually run
            "execution_status": execution_status,
            "user_feedback_rating": user_feedback_rating,
            "calculated_reward": calculated_reward,
            "next_state": next_state, # State for the *next* interaction cycle, can be None
            "done": done, # Typically True for each MasterPlanner cycle
            "timestamp_end_iso": timestamp_end_iso # Logged when R and S' are known
        }

        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json.dump(experience_log_entry, f)
                f.write('\n')
        except IOError as e:
            print(f"ERROR: Could not write to RL experience log file {self.log_file_path}: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected error while logging RL experience: {e}")

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    log_dir_test = Path("./temp_rl_logs")
    log_dir_test.mkdir(exist_ok=True)
    logger = RLExperienceLogger(log_dir_test / "test_rl_experience.jsonl")

    # Dummy data for an interaction
    test_interaction_id = str(uuid.uuid4())
    ts_start = datetime.datetime.now().isoformat()

    # Attempt 0 (initial plan)
    state_0 = {"feature1": 0.5, "nlu_intent": "code_generation", "kb_hits": 3}
    action_0 = "Action_DefaultPrompt"
    prompt_details_0 = {"instruction_variant": "standard"}
    gen_plan_0 = "[{\"step_id\":\"1\", ...}]"
    # Assume this plan was executed

    # Simulate some time passing for execution
    time.sleep(0.1)
    ts_end_0 = datetime.datetime.now().isoformat()

    logger.log_experience(
        rl_interaction_id=test_interaction_id,
        attempt_number=0,
        state=state_0,
        action=action_0,
        master_planner_prompt_details=prompt_details_0,
        generated_plan_json=gen_plan_0,
        plan_parsing_status="success",
        final_executed_plan_json=gen_plan_0, # This attempt's plan was executed
        execution_status="failure", # Say it failed
        user_feedback_rating="negative",
        calculated_reward=-1.25,
        next_state=None, # For this example, next_state is not fully formed yet or is for next interaction
        done=True, # Each planning cycle is an episode
        timestamp_start_iso=ts_start, # When S,A for this attempt was decided
        timestamp_end_iso=ts_end_0    # When outcome of this attempt's execution is known
    )
    print(f"Logged first experience to {logger.log_file_path}")

    # Example for a revision (attempt 1 for the same interaction_id)
    # ts_start_1 = datetime.datetime.now().isoformat() # Start of revision attempt
    # state_1 = {"feature1": 0.6, "nlu_intent": "code_generation", "kb_hits": 3, "prev_attempt_failed": True} # State might include info about previous failure
    # action_1 = "Action_EmphasizeSimplicity"
    # prompt_details_1 = {"added_instruction": "Keep it simple."}
    # gen_plan_1 = "[{\"step_id\":\"1.rev1\", ...}]"
    # time.sleep(0.1)
    # ts_end_1 = datetime.datetime.now().isoformat()

    # logger.log_experience(
    #     rl_interaction_id=test_interaction_id, # Same interaction
    #     attempt_number=1,
    #     state=state_1,
    #     action=action_1,
    #     master_planner_prompt_details=prompt_details_1,
    #     generated_plan_json=gen_plan_1,
    #     plan_parsing_status="success",
    #     final_executed_plan_json=gen_plan_1,
    #     execution_status="success",
    #     user_feedback_rating="positive",
    #     calculated_reward=1.5,
    #     next_state=None,
    #     done=True,
    #     timestamp_start_iso=ts_start_1,
    #     timestamp_end_iso=ts_end_1
    # )
    # print(f"Logged second experience (revision) to {logger.log_file_path}")
