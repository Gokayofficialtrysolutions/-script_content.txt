import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Dict
import datetime
import uuid

class AsyncTaskStatus(Enum):
    """Enumeration of possible statuses for an asynchronous task."""
    PENDING = auto()    # Task has been created but not yet started.
    RUNNING = auto()    # Task is currently being executed.
    COMPLETED = auto()  # Task finished successfully.
    FAILED = auto()     # Task terminated due to an error.
    CANCELLED = auto()  # Task was cancelled before or during execution.

@dataclass
class AsyncTask:
    """
    Represents an asynchronous task managed by the TerminusOrchestrator.
    It holds metadata about the task's execution, status, and results.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the task."""

    name: Optional[str] = None
    """Optional descriptive name for the task (e.g., 'Ollama-AgentX-Summarize')."""

    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    """Current status of the task (e.g., PENDING, RUNNING, COMPLETED)."""

    result: Optional[Any] = None
    """The result of the task if it completed successfully. Structure depends on the task."""

    error: Optional[str] = None
    """Error message or traceback string if the task failed."""

    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    """Timestamp (UTC) when the AsyncTask object was created."""

    started_at: Optional[datetime.datetime] = None
    """Timestamp (UTC) when the task execution actually started."""

    completed_at: Optional[datetime.datetime] = None
    """Timestamp (UTC) when the task finished (either COMPLETED, FAILED, or CANCELLED)."""

    _task_obj: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)
    """
    Internal reference to the actual asyncio.Task object.
    This is primarily for direct manipulation by the orchestrator (e.g., cancellation)
    and is not typically part of serialized representations or external status checks.
    It's excluded from to_dict() and not restored by from_dict().
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the AsyncTask instance to a dictionary suitable for serialization (e.g., JSON).
        The internal `_task_obj` is excluded. Timestamps are converted to ISO 8601 strings.
        """
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.name, # Store enum member name as string
            "result": self.result,     # Note: Result can be complex; further serialization might be needed by consumer.
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncTask':
        """
        Creates an AsyncTask instance from a dictionary representation (e.g., from JSON).
        The internal `_task_obj` is not restored.
        """
        # Ensure created_at defaults to now_utc if missing/None, similar to field default_factory
        created_ts_str = data.get("created_at")
        created_dt = datetime.datetime.fromisoformat(created_ts_str) if created_ts_str else datetime.datetime.utcnow()

        started_ts_str = data.get("started_at")
        started_dt = datetime.datetime.fromisoformat(started_ts_str) if started_ts_str else None

        completed_ts_str = data.get("completed_at")
        completed_dt = datetime.datetime.fromisoformat(completed_ts_str) if completed_ts_str else None

        return cls(
            task_id=data["task_id"], # task_id is mandatory
            name=data.get("name"),
            status=AsyncTaskStatus[data["status"]] if data.get("status") else AsyncTaskStatus.PENDING,
            result=data.get("result"),
            error=data.get("error"),
            created_at=created_dt,
            started_at=started_dt,
            completed_at=completed_dt
            # _task_obj is not restored from dict as it represents a runtime object.
        )

# Example Usage (for testing, not part of the file itself usually)
if __name__ == '__main__':
    # Test with UTC timestamps for consistency
    now_utc = datetime.datetime.utcnow()

    pending_task_info = AsyncTask(name="Test Task Pending", created_at=now_utc)
    print(f"Pending Task Info (dict): {pending_task_info.to_dict()}")

    running_task_info = AsyncTask(name="Test Task Running", status=AsyncTaskStatus.RUNNING,
                                  created_at=now_utc - datetime.timedelta(seconds=1),
                                  started_at=now_utc)
    print(f"Running Task Info (dict): {running_task_info.to_dict()}")

    completed_task_info = AsyncTask(
        name="Test Task Completed",
        status=AsyncTaskStatus.COMPLETED,
        result={"data": "some result", "value": 123},
        created_at=now_utc - datetime.timedelta(seconds=10),
        started_at=now_utc - datetime.timedelta(seconds=5),
        completed_at=now_utc
    )
    print(f"Completed Task Info (dict): {completed_task_info.to_dict()}")
    completed_task_info_from_dict = AsyncTask.from_dict(completed_task_info.to_dict())
    print(f"Restored Completed Task Info: {completed_task_info_from_dict}")

    failed_task_info = AsyncTask(
        name="Test Task Failed",
        status=AsyncTaskStatus.FAILED,
        error="Something went wrong",
        created_at=now_utc - datetime.timedelta(seconds=3),
        started_at=now_utc - datetime.timedelta(seconds=2),
        completed_at=now_utc
    )
    print(f"Failed Task Info (dict): {failed_task_info.to_dict()}")
    failed_task_info_from_dict = AsyncTask.from_dict(failed_task_info.to_dict())
    print(f"Restored Failed Task Info: {failed_task_info_from_dict}")

    assert completed_task_info_from_dict.status == AsyncTaskStatus.COMPLETED, "Status mismatch"
    assert failed_task_info_from_dict.error == "Something went wrong", "Error message mismatch"
    assert pending_task_info.created_at.tzinfo is None # Naive UTC
    assert completed_task_info_from_dict.created_at.tzinfo is None # Naive UTC fromisoformat

    # Test from_dict with missing optional fields
    minimal_data = {"task_id": str(uuid.uuid4()), "status": "COMPLETED"}
    minimal_task = AsyncTask.from_dict(minimal_data)
    print(f"Minimal restored task: {minimal_task.to_dict()}")
    assert minimal_task.status == AsyncTaskStatus.COMPLETED
    assert minimal_task.name is None
    assert minimal_task.created_at is not None # Should default to now_utc

    print("Assertions passed for async_tools.py.")
