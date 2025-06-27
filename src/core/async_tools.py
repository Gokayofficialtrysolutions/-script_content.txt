import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Dict
import datetime
import uuid

class AsyncTaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto() # For future use

@dataclass
class AsyncTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None # Store error message or traceback string
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None

    # To store the actual asyncio.Task object if needed, but primarily for internal tracking by orchestrator
    # This will not be serialized directly if these AsyncTask objects are stored/passed around as pure data.
    _task_obj: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the AsyncTask to a dictionary, excluding the _task_obj."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.name, # Store enum name as string
            "result": self.result, # Result can be complex, handle serialization if needed elsewhere
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncTask':
        """Creates an AsyncTask from a dictionary."""
        return cls(
            task_id=data["task_id"],
            name=data.get("name"),
            status=AsyncTaskStatus[data["status"]] if data.get("status") else AsyncTaskStatus.PENDING,
            result=data.get("result"),
            error=data.get("error"),
            created_at=datetime.datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.datetime.now(),
            started_at=datetime.datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            # _task_obj is not restored from dict
        )

# Example Usage (for testing, not part of the file itself usually)
if __name__ == '__main__':
    pending_task_info = AsyncTask(name="Test Task Pending")
    print(f"Pending Task Info (dict): {pending_task_info.to_dict()}")

    running_task_info = AsyncTask(name="Test Task Running", status=AsyncTaskStatus.RUNNING, started_at=datetime.datetime.now())
    print(f"Running Task Info (dict): {running_task_info.to_dict()}")

    completed_task_info = AsyncTask(
        name="Test Task Completed",
        status=AsyncTaskStatus.COMPLETED,
        result={"data": "some result", "value": 123},
        started_at=datetime.datetime.now() - datetime.timedelta(seconds=5),
        completed_at=datetime.datetime.now()
    )
    print(f"Completed Task Info (dict): {completed_task_info.to_dict()}")
    completed_task_info_from_dict = AsyncTask.from_dict(completed_task_info.to_dict())
    print(f"Restored Completed Task Info: {completed_task_info_from_dict}")

    failed_task_info = AsyncTask(
        name="Test Task Failed",
        status=AsyncTaskStatus.FAILED,
        error="Something went wrong",
        started_at=datetime.datetime.now() - datetime.timedelta(seconds=2),
        completed_at=datetime.datetime.now()
    )
    print(f"Failed Task Info (dict): {failed_task_info.to_dict()}")
    failed_task_info_from_dict = AsyncTask.from_dict(failed_task_info.to_dict())
    print(f"Restored Failed Task Info: {failed_task_info_from_dict}")

    assert completed_task_info_from_dict.status == AsyncTaskStatus.COMPLETED
    assert failed_task_info_from_dict.error == "Something went wrong"
    print("Assertions passed.")
