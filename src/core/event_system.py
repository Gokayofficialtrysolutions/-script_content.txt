import uuid
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SystemEvent:
    """
    Represents a system-level event within the Terminus AGI.
    Events are used for decoupled communication between components (agents, orchestrator modules).
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this specific event instance."""

    event_type: str
    """
    Type of the event, using dot-notation for namespacing (e.g., "kb.content.added",
    "user.feedback.submitted", "agent.task.status_changed").
    """

    source_component: str
    """
    Identifier of the component that published the event
    (e.g., "TerminusOrchestrator.KnowledgeBase", "Agent.WebCrawler", "UserInterface").
    """

    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    """Timestamp (UTC) when the event was created."""

    payload: Dict[str, Any] = field(default_factory=dict)
    """A dictionary containing data specific to this event type."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the SystemEvent instance to a dictionary suitable for serialization (e.g., JSON).
        Timestamp is converted to an ISO 8601 string.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_component": self.source_component,
            "timestamp": self.timestamp.isoformat(), # Ensure UTC if not already timezone-aware
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemEvent':
        """
        Creates a SystemEvent instance from a dictionary representation (e.g., from JSON).
        """
        # Ensure timestamp is parsed correctly; fromisoformat expects tz-aware or assumes naive.
        # Since we store as UTC naive (via default_factory=datetime.datetime.utcnow), parsing as naive is fine.
        ts_str = data.get("timestamp")
        parsed_timestamp = datetime.datetime.fromisoformat(ts_str) if ts_str else datetime.datetime.utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())), # Default to new UUID if missing
            event_type=data["event_type"], # event_type is mandatory
            source_component=data["source_component"], # source_component is mandatory
            timestamp=parsed_timestamp,
            payload=data.get("payload", {}), # payload defaults to empty dict
        )

if __name__ == '__main__':
    # Example Usage
    event1 = SystemEvent(
        event_type="test.event.occurred",
        source_component="ExampleSystem.TestSource",
        payload={"data": "some test data", "value": 123}
    )
    print(f"Event 1 created: ID={event1.event_id}, Timestamp={event1.timestamp.isoformat()}, Source='{event1.source_component}'")
    print(f"Event 1 dict: {event1.to_dict()}")

    event1_dict = event1.to_dict()
    event1_restored = SystemEvent.from_dict(event1_dict)
    print(f"Event 1 restored: ID={event1_restored.event_id}, Payload value: {event1_restored.payload.get('value')}, Timestamp={event1_restored.timestamp.isoformat()}")

    assert event1.event_id == event1_restored.event_id
    assert event1.payload["value"] == event1_restored.payload["value"]
    assert event1.timestamp.replace(microsecond=0) == event1_restored.timestamp.replace(microsecond=0) # Compare without microseconds for robustness

    # Test from_dict with minimal data
    minimal_event_data = {
        "event_type": "minimal.event",
        "source_component": "MinimalSource"
    }
    event2 = SystemEvent.from_dict(minimal_event_data)
    print(f"Event 2 (minimal) created: ID={event2.event_id}, Timestamp={event2.timestamp.isoformat()}")
    assert event2.event_id is not None
    assert event2.payload == {}

    print("SystemEvent basic tests passed.")
