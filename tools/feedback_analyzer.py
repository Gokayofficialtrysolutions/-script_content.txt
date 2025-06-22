import json
import datetime
import argparse
from pathlib import Path
from collections import Counter, defaultdict # Add defaultdict here

def analyze_feedback_log(log_file_path: Path) -> Dict:
    """
    Analyzes a feedback log file (JSONL format) and generates a summary report.
    """
    report = {
        "report_id": str(uuid.uuid4()), # Requires uuid import if not already in this file context
        "report_generation_timestamp_iso": datetime.datetime.now().isoformat(),
        "analysis_source_file": str(log_file_path.name),
        "analysis_period_start_iso": None,
        "analysis_period_end_iso": None,
        "total_feedback_entries_processed": 0,
        "feedback_entries_with_comments": 0,
        "overall_sentiment_counts": defaultdict(int), # Using defaultdict for convenience
        "sentiment_by_item_type": defaultdict(lambda: defaultdict(int)),
        "comment_summary": {
            "total_comments_processed": 0,
            "first_n_comments_preview": [],
            "status_message": "Manual review of all comments in feedback_log.jsonl recommended for detailed insights."
        }
    }

    min_timestamp = None
    max_timestamp = None
    comments_for_preview = []
    MAX_COMMENTS_IN_PREVIEW = 10

    if not log_file_path.exists():
        report["comment_summary"]["status_message"] = f"ERROR: Log file not found at {log_file_path}"
        print(f"ERROR: Log file not found at {log_file_path}")
        return report # Return basic report with error

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    report["total_feedback_entries_processed"] += 1

                    # Update timestamps
                    entry_ts_str = entry.get("timestamp_iso")
                    if entry_ts_str:
                        try:
                            entry_dt = datetime.datetime.fromisoformat(entry_ts_str)
                            if min_timestamp is None or entry_dt < min_timestamp:
                                min_timestamp = entry_dt
                            if max_timestamp is None or entry_dt > max_timestamp:
                                max_timestamp = entry_dt
                        except ValueError:
                            print(f"Warning: Could not parse timestamp '{entry_ts_str}' on line {line_number}. Skipping for period calculation.")

                    rating = entry.get("rating")
                    if rating in ["positive", "negative", "neutral"]:
                        report["overall_sentiment_counts"][rating] += 1

                    item_type = entry.get("item_type", "unknown_item_type")
                    report["sentiment_by_item_type"][item_type]["total_ratings"] += 1
                    if rating in ["positive", "negative", "neutral"]:
                        report["sentiment_by_item_type"][item_type][rating] += 1

                    comment = entry.get("comment", "")
                    if comment and comment.strip():
                        report["feedback_entries_with_comments"] += 1
                        report["comment_summary"]["total_comments_processed"] +=1
                        if len(comments_for_preview) < MAX_COMMENTS_IN_PREVIEW:
                            comments_for_preview.append({
                                "item_id": entry.get("item_id", "N/A"),
                                "item_type": item_type,
                                "rating": rating,
                                "comment": comment[:200] + "..." if len(comment) > 200 else comment # Preview comment
                            })

                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_number}: {line.strip()[:100]}...")
                except Exception as e_inner:
                    print(f"Warning: Error processing line {line_number}: {e_inner}. Line: {line.strip()[:100]}...")

        if min_timestamp:
            report["analysis_period_start_iso"] = min_timestamp.isoformat()
        if max_timestamp:
            report["analysis_period_end_iso"] = max_timestamp.isoformat()

        report["comment_summary"]["first_n_comments_preview"] = comments_for_preview

    except FileNotFoundError:
        # This is already handled by the check at the beginning, but as a safeguard.
        report["comment_summary"]["status_message"] = f"ERROR: Log file not found at {log_file_path}"
        print(f"ERROR: Log file not found at {log_file_path}")
    except Exception as e:
        error_msg = f"ERROR: Failed to process feedback log file {log_file_path}. Error: {str(e)}"
        report["comment_summary"]["status_message"] = error_msg
        print(error_msg)

    # Convert defaultdicts to dicts for clean JSON output
    report["overall_sentiment_counts"] = dict(report["overall_sentiment_counts"])
    report["sentiment_by_item_type"] = {k: dict(v) for k, v in report["sentiment_by_item_type"].items()}

    return report

def main():
    parser = argparse.ArgumentParser(description="Analyzes user feedback log and generates a summary report.")
    parser.add_argument(
        "--log_file",
        type=str,
        default="../logs/feedback_log.jsonl", # Default relative to tools/ directory
        help="Path to the feedback_log.jsonl file."
    )
    args = parser.parse_args()

    # Resolve the path relative to this script's location if it's a relative path
    script_dir = Path(__file__).parent
    log_file_path = Path(args.log_file)
    if not log_file_path.is_absolute():
        log_file_path = (script_dir / log_file_path).resolve()

    # Need to import uuid here for report_id if it's not globally available
    # For now, assuming uuid is imported where this script is called or this function is used.
    # If run standalone, it will need 'import uuid' at the top of the file.
    # Let's add it for standalone robustness.
    global uuid # To assign to the global uuid if not already imported (for the report_id)
    import uuid # Ensure uuid is imported for report_id generation

    summary_report = analyze_feedback_log(log_file_path)
    print(json.dumps(summary_report, indent=2))

if __name__ == "__main__":
    main()
