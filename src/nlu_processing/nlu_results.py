from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

class Intent(BaseModel):
    name: str = Field(..., description="Name of the detected intent.")
    confidence: float = Field(..., description="Confidence score for the detected intent (0.0 to 1.0).")
    alternate_intents: Optional[List[Tuple[str, float]]] = Field(None, description="List of (name, confidence) for other possible intents.")

class Entity(BaseModel):
    text: str = Field(..., description="The actual text of the entity.")
    label: str = Field(..., description="The entity type or label (e.g., 'PERSON', 'ORG', 'FILE_PATH').")
    start_char: int = Field(..., description="Start character offset in the raw text.")
    end_char: int = Field(..., description="End character offset in the raw text.")
    value: Optional[Any] = Field(None, description="Normalized value for the entity, if applicable (e.g., a standardized date string for a date entity).")

class Sentiment(BaseModel):
    label: str = Field(..., description="Sentiment label (e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL').")
    score: float = Field(..., description="Sentiment score, specific to the model used.")

class ParsedCommand(BaseModel):
    command: str = Field(..., description="The recognized command or action.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameters and their values for the command.")

class NLUResult(BaseModel):
    raw_text: str = Field(..., description="The original input text.")
    detected_intent: Optional[Intent] = Field(None, description="The primary detected intent.")
    entities: List[Entity] = Field(default_factory=list, description="List of detected entities.")
    sentiment: Optional[Sentiment] = Field(None, description="Detected sentiment of the text.")
    parsed_command: Optional[ParsedCommand] = Field(None, description="Structured command parsed from the text, if applicable.")
    language: str = Field("en", description="Detected language of the input text (ISO 639-1 code).")
    processing_time_ms: Optional[float] = Field(None, description="Time taken for NLU processing in milliseconds.")
    tokens: Optional[List[str]] = Field(None, description="List of tokens from the input text.")
    sentences: Optional[List[str]] = Field(None, description="List of sentences from the input text.")

    class Config:
        json_encoders = {
            # If you have custom types that need special JSON encoding
        }
        # For example, if using Pydantic V2
        # model_config = {"json_encoders": {}}

    def get_entities_by_label(self, label: str) -> List[Entity]:
        """Helper to filter entities by a specific label."""
        return [entity for entity in self.entities if entity.label == label]

    def __str__(self):
        intent_str = f"Intent: {self.detected_intent.name} (Confidence: {self.detected_intent.confidence:.2f})" if self.detected_intent else "Intent: None"
        entities_str = f"Entities: {len(self.entities)}"
        if self.entities:
            entities_str += "\n  " + "\n  ".join([f"{e.text} ({e.label})" for e in self.entities])
        sentiment_str = f"Sentiment: {self.sentiment.label} (Score: {self.sentiment.score:.2f})" if self.sentiment else "Sentiment: None"
        return f"NLUResult for \"{self.raw_text[:50]}...\":\n  {intent_str}\n  {entities_str}\n  {sentiment_str}"
