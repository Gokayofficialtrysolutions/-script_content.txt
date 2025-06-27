import uuid
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SystemEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str # e.g., "kb.content.added", "user.feedback.submitted"
    source_component: str # e.g., "TerminusOrchestrator.KB", "Agent.WebCrawler"
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow) # Use UTC
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_component": self.source_component,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemEvent':
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            source_component=data["source_component"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
        )

if __name__ == '__main__':
    # Example Usage
    event1 = SystemEvent(
        event_type="test.event.occurred",
        source_component="ExampleSystem.TestSource",
        payload={"data": "some test data", "value": 123}
    )
    print(f"Event 1 created: {event1.event_id} at {event1.timestamp} from {event1.source_component}")
    print(f"Event 1 dict: {event1.to_dict()}")

    event1_dict = event1.to_dict()
    event1_restored = SystemEvent.from_dict(event1_dict)
    print(f"Event 1 restored: {event1_restored.event_id}, Payload value: {event1_restored.payload.get('value')}")
    assert event1.event_id == event1_restored.event_id
    assert event1.payload["value"] == event1_restored.payload["value"]

    print("SystemEvent basic tests passed.")
