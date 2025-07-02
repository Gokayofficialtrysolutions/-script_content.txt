import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import logging
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from .nlu_results import NLUResult, Intent, Entity, Sentiment, ParsedCommand
from .nlu_config import NLUConfig, nlu_default_config

# Configure logging
logger = logging.getLogger(__name__)
# Example: logging.basicConfig(level=logging.INFO) # Configure in main application

# Custom Exception for NLU Processing
class NLUProcessingError(Exception):
    pass

class NLUProcessor:
    def __init__(self, config: NLUConfig = nlu_default_config):
        self.config = config
        self.nlp = None
        self.intent_pipeline = None
        self.sentiment_pipeline = None
        self.matcher = None
        self.command_rules = []

        self._load_models()
        if self.config.enable_rule_based_entities and self.config.command_patterns_file:
            self._load_command_patterns(self.config.command_patterns_file)

    def _load_models(self):
        logger.info(f"Loading spaCy model: {self.config.spacy_model_name}")
        try:
            self.nlp = spacy.load(self.config.spacy_model_name)
            self.matcher = Matcher(self.nlp.vocab)
        except OSError as e:
            logger.error(f"Could not load spaCy model '{self.config.spacy_model_name}'. "
                         f"Ensure it's downloaded (e.g., python -m spacy download {self.config.spacy_model_name}). Error: {e}")
            raise NLUProcessingError(f"Failed to load spaCy model: {self.config.spacy_model_name}") from e

        # Initialize coreferee if enabled and model supports it (example)
        # if self.config.enable_coreference_resolution:
        #     try:
        #         if 'coreferee' not in self.nlp.pipe_names: # Check if already added
        #             # Ensure coreferee is installed: pip install coreferee
        #             # Ensure compatible models are downloaded: python -m coreferee install en
        #             import coreferee
        #             self.nlp.add_pipe('coreferee')
        #             logger.info("coreferee pipe added to spaCy model for co-reference resolution.")
        #         else:
        #             logger.info("coreferee pipe already present in spaCy model.")
        #     except ImportError:
        #         logger.error("coreferee library not installed. Co-reference resolution will be disabled. pip install coreferee")
        #         self.config.enable_coreference_resolution = False
        #     except Exception as e_coref:
        #         logger.error(f"Error initializing coreferee: {e_coref}. Co-reference resolution disabled.")
        #         self.config.enable_coreference_resolution = False


        logger.info(f"Loading Intent Recognition model: {self.config.intent_model_name}")
        try:
            # Using a generic text classification model for now.
            # Replace with a model fine-tuned for specific intents if available.
            # For multi-label intents, you might need a different model setup.
            self.intent_pipeline = pipeline(
                "text-classification",
                model=self.config.intent_model_name,
                tokenizer=self.config.intent_model_name,
                # top_k=self.config.max_intent_alternates # Get multiple results if model supports
            )
        except Exception as e:
            logger.error(f"Could not load Intent Recognition model '{self.config.intent_model_name}'. Error: {e}")
            # Decide behavior: raise error or run without intent recognition.
            # For now, we'll allow it to proceed without intent if it fails.
            self.intent_pipeline = None
            logger.warning("Intent recognition will be unavailable.")

        if self.config.enable_sentiment_analysis:
            logger.info(f"Loading Sentiment Analysis model: {self.config.sentiment_model_name}")
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.config.sentiment_model_name,
                    tokenizer=self.config.sentiment_model_name
                )
            except Exception as e:
                logger.error(f"Could not load Sentiment Analysis model '{self.config.sentiment_model_name}'. Error: {e}")
                self.sentiment_pipeline = None # Allow proceeding without sentiment
                logger.warning("Sentiment analysis will be unavailable.")
        else:
            logger.info("Sentiment analysis is disabled by configuration.")
            self.sentiment_pipeline = None

    def _load_command_patterns(self, patterns_file_path: str):
        """Loads command patterns from a JSON file and adds them to the spaCy Matcher."""
        if not self.matcher:
            logger.error("Matcher not initialized. Cannot load command patterns.")
            return

        try:
            # Resolve patterns_file_path relative to this file's directory if it's not absolute
            resolved_path = Path(__file__).parent / patterns_file_path
            if not resolved_path.is_file():
                # Try as absolute path if relative failed
                resolved_path = Path(patterns_file_path)

            if not resolved_path.is_file():
                logger.error(f"Command patterns file not found at '{patterns_file_path}' (also checked {resolved_path}).")
                return

            with open(resolved_path, 'r', encoding='utf-8') as f:
                self.command_rules = json.load(f)

            for rule_config in self.command_rules:
                command_name = rule_config["command_name"]
                patterns = rule_config["patterns"]
                self.matcher.add(command_name, patterns) # patterns is list of lists of dicts
                # Store entity mappings and default params if needed later by _parse_command
            logger.info(f"Loaded {len(self.command_rules)} command rules with {sum(len(r['patterns']) for r in self.command_rules)} patterns from '{resolved_path}'.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from command patterns file '{resolved_path}': {e}")
            self.command_rules = []
        except Exception as e:
            logger.error(f"Error loading command patterns from '{resolved_path}': {e}")
            self.command_rules = []


    def _extract_spacy_entities(self, doc: Doc) -> List[Entity]:
        entities = []
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char
            ))
        return entities

    def _recognize_intent(self, text: str, doc: Optional[Doc] = None) -> Optional[Intent]:
        """
        Recognizes intent. If doc is provided, it might also check for command patterns
        that imply an intent.
        """
        # First, try LLM-based intent recognition
        llm_intent: Optional[Intent] = None
        if self.intent_pipeline:
            try:
                predictions = self.intent_pipeline(text, top_k=self.config.max_intent_alternates if self.config.max_intent_alternates > 0 else 1)
                if predictions:
                    if not isinstance(predictions, list): predictions = [predictions]
                    primary_prediction = predictions[0]
                    primary_intent_name = primary_prediction['label']
                    primary_intent_confidence = primary_prediction['score']

                    if primary_intent_confidence >= self.config.intent_confidence_threshold:
                        alternates = []
                        if len(predictions) > 1:
                            for alt_pred in predictions[1:]: alternates.append((alt_pred['label'], alt_pred['score']))
                        llm_intent = Intent(name=primary_intent_name, confidence=primary_intent_confidence, alternate_intents=alternates if alternates else None)
                    else:
                        logger.info(f"LLM Intent '{primary_intent_name}' confidence {primary_intent_confidence:.2f} below threshold {self.config.intent_confidence_threshold}.")
            except Exception as e:
                logger.error(f"Error during LLM intent recognition for text '{text[:50]}...': {e}", exc_info=True)

        # If command parsing is enabled and doc is available, check if a command pattern implies an intent
        # This part is more about command parsing, but a detected command could override or inform intent.
        # For now, intent recognition is primarily LLM based. Command parsing will populate `parsed_command`.
        # A sophisticated system might have command patterns directly map to intents.
        # For example, if a "CREATE_FILE" command is detected, the intent might be set to "CREATE_FILE_INTENT".
        # This logic can be added here if command rules define `intent_aliases`.

        # Simple strategy: return LLM intent for now. Command parsing is separate.
        # More advanced: if a command is parsed, and that command has an associated intent in command_rules,
        # potentially use that if its "confidence" (based on match quality) is higher or LLM intent is weak.
        return llm_intent


    def _parse_command_from_doc(self, doc: Doc) -> Optional[ParsedCommand]:
        """Uses spaCy Matcher to find command patterns in the doc and extracts parameters."""
        if not self.matcher or not self.command_rules:
            return None

        matches = self.matcher(doc)
        if not matches:
            return None

        # Simple strategy: use the first match. Could be more sophisticated (e.g., longest match).
        # Sort matches by start token and then by length (longest match first for overlapping patterns)
        # Spacy matches are (match_id, start_token_idx, end_token_idx)
        # We need to map match_id back to our rule_config to get command_name and entity_mapping

        best_match = None
        # Prioritize longer matches if they start at the same token, or earlier matches.
        # Spacy's default match order might be sufficient for many cases.
        # For this example, let's just take the first one found.
        # A more robust system might score matches or use other heuristics.

        # For simplicity, let's just use the first match found by spaCy's matcher.
        # spaCy's matcher itself doesn't guarantee order of multiple patterns matching at same spot.
        # If multiple rules match, we might need a priority system.

        # Iterate through spaCy's matches
        for match_id, start, end in sorted(matches, key=lambda m: (m[1], m[2] - m[1]), reverse=True): # Longest match first
            command_name_hash = match_id # This is the hash of the rule name string
            matched_span = doc[start:end]

            # Find the original rule_config that corresponds to this match_id
            matched_rule_config = None
            for rule_cfg in self.command_rules:
                # spaCy stores match_id as a hash of the string rule name. We need to compare.
                # This is a bit of a hack. spaCy's Matcher returns the hash.
                # A better way is to store a mapping from hash to rule_config when adding rules.
                # For now, we'll assume `matcher.vocab.strings[match_id]` gives back the string name.
                rule_name_from_vocab = self.nlp.vocab.strings[match_id]
                if rule_cfg["command_name"] == rule_name_from_vocab:
                    matched_rule_config = rule_cfg
                    break

            if not matched_rule_config:
                logger.warning(f"Could not find rule config for match_id {match_id} ('{self.nlp.vocab.strings[match_id]}'). Skipping this match.")
                continue

            logger.info(f"Found command pattern match for '{matched_rule_config['command_name']}' text: '{matched_span.text}'")

            parameters: Dict[str, Any] = {}
            # Default parameters from rule config
            parameters.update(matched_rule_config.get("default_params", {}))

            # Extract entities based on "CAPTURE" keys in the spaCy pattern
            # This requires that the spaCy patterns use the "OP": "CAPTURE" mechanism
            # which is not standard spaCy Matcher syntax.
            # spaCy's matcher captures based on the name given to a token pattern with an operator.
            # Example: [{"LOWER": "file"}, {"ENT_TYPE": "FILENAME", "OP": "+", "_": {"capture": "my_filename"}}]
            # This is complex. For now, let's assume entities are extracted by spaCy's NER
            # and we map them based on the `entity_mapping` in our JSON rules.

            # Simpler approach for now: iterate over defined entity_mappings in the rule
            # and try to find those entities within the matched_span via regular NER or specific rules.
            # This is less precise than direct capture from matcher patterns.

            # Let's refine this: the `patterns` in JSON should align with spaCy's pattern structure.
            # The "CAPTURE" key is conceptual for our JSON. We need to map it to spaCy's way or process spans.

            # Corrected approach:
            # The spaCy Matcher itself doesn't easily return named captures from complex patterns
            # in the way one might expect from regex. When a pattern in `matcher.add` matches,
            # you get the `match_id` and the `span`.
            # To get "captured" parts, you often re-process the `matched_span` or design patterns
            # carefully so specific tokens/entities can be identified by their role in the match.

            # For this implementation, let's assume `entity_mapping` refers to entity labels
            # that spaCy's NER would find within the `matched_span`, or it refers to a specific
            # token index logic (which is harder).

            # A pragmatic approach for "CAPTURE" in our JSON:
            # The rule definition (`command_rules`) contains `entity_mapping`.
            # e.g., "entity_mapping": { "file_path_token_text": "target_path" }
            # This means if a token within the matched span was marked with "CAPTURE": "file_path_token_text"
            # in the JSON pattern, its text should be assigned to the "target_path" parameter.
            # This requires parsing our custom "CAPTURE" directive.

            # Let's simplify: Iterate over spaCy entities within the matched_span.
            # If an entity's label is in our `entity_mapping` values, we might use it.
            # This is still not ideal.

            # Correct logic for spaCy Matcher and capturing:
            # When adding patterns, the "id" of the pattern part is the capture name.
            # Example pattern for matcher.add: `[{"LOWER": "hello"}, {"IS_PUNCT": True, "id": "punct_capture"}]`
            # Then, `on_match` callback receives `matcher, doc, i, matches`.
            # `matches[i]` is `(match_id, start, end)`. `doc[start:end]` is the full span.
            # To get sub-spans for captures, you need to know which part of your pattern was named.
            # This is typically done by having multiple, more granular patterns or a complex on_match.

            # Given our current JSON structure for patterns, direct capture is hard.
            # Fallback: Use NER entities within the span and map them.
            custom_captured_entities = {} # For our conceptual "CAPTURE"

            # This is a placeholder for a more robust capture mechanism.
            # For now, we will rely on the `entity_mapping` to map NER results from the matched span.
            for ent_in_span in matched_span.ents:
                for map_key_in_rule, param_name_in_output in matched_rule_config.get("entity_mapping", {}).items():
                    # This mapping is tricky. `map_key_in_rule` needs to relate to how `ent_in_span.label_` or `ent_in_span.text` is identified.
                    # Let's assume `map_key_in_rule` is the expected `ent_in_span.label_` for now.
                    if ent_in_span.label_ == map_key_in_rule:
                         parameters[param_name_in_output] = ent_in_span.text
                         logger.debug(f"Command '{matched_rule_config['command_name']}': Mapped entity '{ent_in_span.text}' ({ent_in_span.label_}) to param '{param_name_in_output}'")

            # If no entities were mapped, but the pattern structure implies captures, this needs enhancement.
            # For example, if pattern was `[{"LOWER": "file"}, {"SHAPE": "xxxx", "CAPTURE": "filename"}]`
            # We'd need to iterate tokens in `matched_span` and check their attributes.

            # For now, this is a simplified command parser.
            # A truly robust one would require careful pattern design and possibly custom on_match callbacks.
            if parameters: # Only return if we successfully mapped some parameters or have defaults
                 return ParsedCommand(command=matched_rule_config["command_name"], parameters=parameters)
            else:
                logger.debug(f"Command '{matched_rule_config['command_name']}' matched, but no parameters were extracted from span '{matched_span.text}' based on current mapping logic.")

        return None # No suitable match found or no params extracted


    def _analyze_sentiment(self, text: str) -> Optional[Sentiment]:
        if not self.sentiment_pipeline:
            return None

        try:
            # The default text-classification pipeline might not return multiple labels easily.
            # For now, we get the top one. If a multi-label model is used, adjust accordingly.
            # Some pipelines return a list of dicts, others a single dict.
            # The 'distilbert-base-uncased-finetuned-sst-2-english' model is for sentiment (POSITIVE/NEGATIVE)
            # so it's not ideal for general intent. This is a placeholder.
            # A true intent model would have labels like "CREATE_FILE", "ASK_QUESTION", etc.

            # HACK: Using sst-2 model as a placeholder. It will output POSITIVE/NEGATIVE.
            # This needs to be replaced with a proper intent model.
            # For now, we'll just use the label it gives as the "intent name".
            predictions = self.intent_pipeline(text, top_k=self.config.max_intent_alternates if self.config.max_intent_alternates > 0 else 1)

            if not predictions:
                return None

            # Assuming predictions is a list of dicts like [{'label': 'LABEL_X', 'score': 0.9}, ...]
            # or a single dict if top_k=1 (though often it's still a list with one item)
            if not isinstance(predictions, list):
                predictions = [predictions]

            primary_prediction = predictions[0]
            primary_intent_name = primary_prediction['label']
            primary_intent_confidence = primary_prediction['score']

            if primary_intent_confidence < self.config.intent_confidence_threshold:
                logger.info(f"Intent '{primary_intent_name}' confidence {primary_intent_confidence:.2f} is below threshold {self.config.intent_confidence_threshold}.")
                return None # Or return with a special "UNCLEAR_INTENT"

            alternates = []
            if len(predictions) > 1:
                for alt_pred in predictions[1:]:
                    alternates.append((alt_pred['label'], alt_pred['score']))

            return Intent(
                name=primary_intent_name,
                confidence=primary_intent_confidence,
                alternate_intents=alternates if alternates else None
            )
        except Exception as e:
            logger.error(f"Error during intent recognition for text '{text[:50]}...': {e}", exc_info=True)
            return None


    def _analyze_sentiment(self, text: str) -> Optional[Sentiment]:
        if not self.sentiment_pipeline:
            return None
        try:
            # Sentiment pipeline usually returns a list with a single dict: [{'label': 'POSITIVE', 'score': 0.99}]
            result = self.sentiment_pipeline(text)
            if result and isinstance(result, list):
                sentiment_data = result[0]
                return Sentiment(label=sentiment_data['label'], score=sentiment_data['score'])
            elif result and isinstance(result, dict): # Some pipelines might return a dict directly
                 return Sentiment(label=result['label'], score=result['score'])
            return None
        except Exception as e:
            logger.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}", exc_info=True)
            return None

    def process_text(self, text: str, context: Optional[dict] = None) -> NLUResult: # Placeholder for context
        if not self.nlp:
            raise NLUProcessingError("NLUProcessor is not initialized properly (spaCy model missing).")

        start_time = time.time()

        # SpaCy processing
        doc = self.nlp(text)

        # Core NLU capabilities (Phase 1)
        spacy_entities = self._extract_spacy_entities(doc)

        # Intent Recognition (using placeholder model for now)
        # Pass the doc to _recognize_intent if it might use matcher results for intent refinement later
        detected_intent = self._recognize_intent(text, doc=doc)

        # Sentiment Analysis (if enabled)
        detected_sentiment = None
        if self.config.enable_sentiment_analysis:
            detected_sentiment = self._analyze_sentiment(text)

        # Command Parsing (Phase 2)
        parsed_command = None
        if self.config.enable_rule_based_entities: # Using this flag for command parsing as well
            parsed_command = self._parse_command_from_doc(doc)
            if parsed_command:
                logger.info(f"Parsed command: {parsed_command.command} with params: {parsed_command.parameters}")
                # Optional: If a command is parsed, and it has an associated intent_alias,
                # we could refine `detected_intent` here.
                # Example: if parsed_command.command == "CREATE_FILE" and "CREATE_FILE_INTENT" is an alias,
                # and if llm_intent is weak or absent, set detected_intent to "CREATE_FILE_INTENT".
                # This requires command_rules to include 'intent_aliases'.
                for rule_cfg in self.command_rules:
                    if rule_cfg["command_name"] == parsed_command.command:
                        intent_aliases = rule_cfg.get("intent_aliases", [])
                        if intent_aliases:
                            # Simple strategy: if LLM intent is weak or different, and command implies a strong intent
                            if not detected_intent or detected_intent.confidence < 0.7:
                                # Use the first alias as the intent, with high confidence
                                # A more sophisticated approach would be needed for multiple aliases or confidence scoring.
                                new_intent_name = intent_aliases[0]
                                if not detected_intent or detected_intent.name != new_intent_name:
                                    logger.info(f"Command parsing suggests intent '{new_intent_name}', overriding/setting LLM intent ({detected_intent.name if detected_intent else 'None'}).")
                                    detected_intent = Intent(name=new_intent_name, confidence=0.95) # Assume high confidence for pattern match
                            elif detected_intent.name not in intent_aliases and detected_intent.confidence >= 0.7:
                                # LLM intent is confident and different from command-implied intents.
                                # Could log a warning or add command-implied intent as an alternative.
                                logger.info(f"LLM intent '{detected_intent.name}' is confident but differs from command-implied intents {intent_aliases} for command '{parsed_command.command}'. Keeping LLM intent.")
                        break


        # Co-reference Resolution (Future - Placeholder)
        # if self.config.enable_coreference_resolution and doc._.coref_chains:
        #    resolved_text = doc._.coref_chains.resolve_coref_chains_text(doc)
        #    logger.info(f"Coreference resolved text: {resolved_text[:100]}...")
        #    # This resolved_text could be used for further processing or returned.
        #    # For now, we are not modifying the main NLUResult based on this.

        # Token and sentence extraction
        tokens = [token.text for token in doc]
        sentences = [sent.text for sent in doc.sents]

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        return NLUResult(
            raw_text=text,
            detected_intent=detected_intent,
            entities=spacy_entities,
            sentiment=detected_sentiment,
            parsed_command=parsed_command, # Add parsed command
            language=doc.lang_ if doc.lang_ else self.config.default_language, # spaCy doc.lang_
            processing_time_ms=processing_time_ms,
            tokens=tokens,
            sentences=sentences
        )

# Example Usage (for testing purposes, typically not here)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configure to use a smaller model for quicker testing if needed
    # test_config = NLUConfig(spacy_model_name="en_core_web_sm",
    #                         intent_model_name="distilbert-base-uncased-finetuned-sst-2-english", # Placeholder
    #                         sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english")

    # Using default config which specifies en_core_web_md
    # Ensure en_core_web_md is downloaded: python -m spacy download en_core_web_md

    logger.info("Initializing NLUProcessor with default config...")
    try:
        processor = NLUProcessor() # Uses nlu_default_config

        test_text_1 = "Can you create a Python script to list all files in the /tmp directory? I'm feeling great about this."
        logger.info(f"\nProcessing text 1: \"{test_text_1}\"")
        result_1 = processor.process_text(test_text_1)
        print(result_1)
        print(f"  Tokens: {result_1.tokens[:10]}...")
        print(f"  Sentences: {result_1.sentences}")
        if result_1.parsed_command:
            print(f"  Parsed Command: {result_1.parsed_command.command} with params {result_1.parsed_command.parameters}")


        test_text_2 = "Analyze the sentiment of this: I am very unhappy with the CodeMaster agent's performance on the last task."
        logger.info(f"\nProcessing text 2: \"{test_text_2}\"")
        result_2 = processor.process_text(test_text_2)
        print(result_2)
        if result_2.parsed_command:
            print(f"  Parsed Command: {result_2.parsed_command.command} with params {result_2.parsed_command.parameters}")

        test_text_3 = "What is the weather like in London today?" # Should not match a command
        logger.info(f"\nProcessing text 3: \"{test_text_3}\"")
        result_3 = processor.process_text(test_text_3)
        print(result_3)
        if result_3.entities:
            for ent in result_3.entities:
                print(f"  Entity: {ent.text} ({ent.label_})") # Changed from ent.label to ent.label_ for spaCy v3
        if result_3.parsed_command:
            print(f"  Parsed Command: {result_3.parsed_command.command} with params {result_3.parsed_command.parameters}")


        # Test with intent model (placeholder, will give POSITIVE/NEGATIVE)
        test_text_intent = "I want to generate a new image of a cat."
        logger.info(f"\nProcessing text for intent (placeholder): \"{test_text_intent}\"")
        result_intent = processor.process_text(test_text_intent)
        print(result_intent)
        if result_intent.parsed_command:
            print(f"  Parsed Command: {result_intent.parsed_command.command} with params {result_intent.parsed_command.parameters}")

        # Test a command parsing example
        test_text_command = "Create a file named my_new_document.txt"
        logger.info(f"\nProcessing command text: \"{test_text_command}\"")
        result_command = processor.process_text(test_text_command)
        print(result_command) # This will now call the __str__ of NLUResult
        if result_command.parsed_command:
            print(f"  COMMAND DETECTED: {result_command.parsed_command.command}")
            print(f"  PARAMETERS: {result_command.parsed_command.parameters}")
        else:
            print("  NO COMMAND DETECTED.")
            if result_command.entities:
                 print(f"  Entities found: {[e.text for e in result_command.entities]}")


    except NLUProcessingError as e:
        logger.error(f"NLU Processing Error during example usage: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during example usage: {e}", exc_info=True)

    # Test case where intent model might be missing (if you rename one in config temporarily)
    # test_config_broken_intent = NLUConfig(intent_model_name="non_existent_model_path")
    # try:
    #     logger.info("\nInitializing NLUProcessor with a potentially broken intent model config...")
    #     broken_processor = NLUProcessor(config=test_config_broken_intent)
    #     res_broken = broken_processor.process_text("This should still work for entities.")
    #     print(res_broken)
    # except Exception as e:
    #     logger.error(f"Error with broken config: {e}")

"""
TODO for Phase 1 Completion:
1.  [x] Basic Module Structure: `NLUProcessor` class, load spaCy.
2.  [x] Core Entity Extraction: Implement `_extract_spacy_entities` using `doc.ents`.
3.  [x] Placeholder Intent Recognition:
    *   [x] Integrate Hugging Face `pipeline` for text classification.
    *   [x] Use a generic model like `distilbert-base-uncased-finetuned-sst-2-english` as a STAND-IN.
        (This means "intents" will be 'POSITIVE'/'NEGATIVE' for now, which is not true intent, but tests the mechanism).
    *   [x] Connect to `NLUResult.detected_intent`.
    *   [x] Add confidence threshold from config.
4.  [ ] Unit Tests:
    *   Test entity extraction (e.g., dates, locations, person names from spaCy).
    *   Test placeholder intent recognition (ensure it calls the pipeline and populates the field).
    *   Test sentiment analysis (if enabled).
    *   Test basic NLUResult structure.
"""
