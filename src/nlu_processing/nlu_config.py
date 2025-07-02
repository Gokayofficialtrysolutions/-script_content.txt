from pydantic import BaseModel, Field
from typing import Optional

class NLUConfig(BaseModel):
    # SpaCy settings
    spacy_model_name: str = Field("en_core_web_md", description="Name of the spaCy model to use (e.g., en_core_web_sm, en_core_web_md, en_core_web_lg).")

    # Hugging Face Transformers settings for Intent Recognition
    intent_model_name: str = Field("distilbert-base-uncased-finetuned-sst-2-english",
                                   description="Name or path of the Hugging Face model for intent recognition. Replace with a dedicated intent model.")
    intent_confidence_threshold: float = Field(0.6, description="Minimum confidence score to accept a detected intent.")
    max_intent_alternates: int = Field(3, description="Maximum number of alternative intents to return.")

    # Hugging Face Transformers settings for Sentiment Analysis
    # Using a generic sentiment model for now, can be replaced with a more specific one.
    sentiment_model_name: str = Field("distilbert-base-uncased-finetuned-sst-2-english",
                                      description="Name or path of the Hugging Face model for sentiment analysis.")

    # Feature flags
    enable_sentiment_analysis: bool = Field(True, description="Flag to enable/disable sentiment analysis.")
    enable_rule_based_entities: bool = Field(True, description="Flag to enable/disable custom rule-based entity matching.") # Enable for command parsing
    enable_coreference_resolution: bool = Field(False, description="Flag to enable/disable co-reference resolution.") # Keep False for now

    # Performance
    default_language: str = Field("en", description="Default language if detection fails or is not implemented.")

    # Co-reference settings (using coreferee as an example, if integrated)
    # coreference_model_name: Optional[str] = Field(None, description="Name or path of the model for co-reference resolution. e.g., for coreferee, this might be implicitly tied to spacy model.")

    # Command Parsing settings
    command_patterns_file: Optional[str] = Field("nlu_command_patterns.json", description="Filename (relative to nlu_processing module path or absolute) for rule-based command patterns.")

# Global configuration instance (can be loaded from a file or environment variables in a real app)
# For simplicity, we'll use a default instance here.
# In a larger application, you might use a dependency injection framework or a global config loader.
nlu_default_config = NLUConfig()

# Example of how you might load from a YAML or JSON file:
# import yaml
# def load_config_from_file(filepath: str) -> NLUConfig:
#     with open(filepath, 'r') as f:
#         data = yaml.safe_load(f)
#     return NLUConfig(**data.get("nlu_config", {}))

# nlu_default_config = load_config_from_file("config.yaml") # Example
