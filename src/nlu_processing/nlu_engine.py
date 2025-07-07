import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import logging
import json
from pathlib import Path
from spacy.matcher import Matcher # Restored import
from typing import Optional, List, Tuple, Dict, Any
import inspect

from .nlu_results import NLUResult, Intent, Entity, Sentiment, ParsedCommand
from .nlu_config import NLUConfig, nlu_default_config

# Configure logging
logger = logging.getLogger(__name__)

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
        logger.info(f"Attempting to load models for NLUProcessor...")
        try:
            logger.info(f"Loading spaCy model: {self.config.spacy_model_name}")
            self.nlp = spacy.load(self.config.spacy_model_name)
            logger.info(f"spaCy model {self.config.spacy_model_name} loaded successfully. self.nlp: {self.nlp}")

            # # Add EntityRuler for OBJECTIVE_ID (Temporarily commented out for debugging Matcher)
            # try:
            #     logger.info("Attempting to set up EntityRuler for OBJECTIVE_ID...")
            #     if self.nlp and "entity_ruler" not in self.nlp.pipe_names:
            #         ruler = self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
            #         logger.info("Added new EntityRuler pipe.")
            #     else:
            #         if self.nlp:
            #             ruler = self.nlp.get_pipe("entity_ruler")
            #             logger.info("Retrieved existing EntityRuler pipe.")
            #         else:
            #             logger.error("self.nlp is None before attempting to get EntityRuler.")
            #             raise NLUProcessingError("spaCy nlp object is None, cannot proceed with EntityRuler.")
            #
            #     obj_id_patterns = [
            #         {
            #             "label": "OBJECTIVE_ID",
            #             "pattern": [{"TEXT": {"REGEX": "^userobj_[0-9a-fA-F\\-]+$"}}]
            #         }
            #     ]
            #     if ruler:
            #         ruler.add_patterns(obj_id_patterns)
            #         logger.info("Successfully added EntityRuler patterns for OBJECTIVE_ID.")
            #     else:
            #         logger.error("EntityRuler component (ruler) is None. Cannot add patterns.")
            #         raise NLUProcessingError("EntityRuler component could not be initialized or retrieved.")
            #
            # except Exception as e_ruler:
            #     logger.error(f"Failed during EntityRuler setup for OBJECTIVE_ID: {e_ruler}", exc_info=True)
            #     raise NLUProcessingError(f"Critical failure during EntityRuler setup: {e_ruler}") from e_ruler
            logger.info("EntityRuler setup temporarily skipped.") # Simplified log

            # ADDED LOGGING
            logger.info(f"Pre-Matcher check: self.nlp is {self.nlp}")
            if self.nlp:
                logger.info(f"Pre-Matcher check: type(self.nlp) is {type(self.nlp)}")
                logger.info(f"Pre-Matcher check: self.nlp.vocab is {self.nlp.vocab}")
                logger.info(f"Pre-Matcher check: type(self.nlp.vocab) is {type(self.nlp.vocab)}")
                logger.info(f"Pre-Matcher check: Condition (self.nlp and self.nlp.vocab) is {bool(self.nlp and self.nlp.vocab)}")
            else:
                logger.warning("Pre-Matcher check: self.nlp is None or invalid.")

            logger.info(f"Initializing spaCy Matcher with vocab from: {self.nlp}") # This log was already there
            if self.nlp and self.nlp.vocab: # Ensure nlp and vocab are valid
                logger.info(f"Attempting to create Matcher with vocab: {self.nlp.vocab}")

                # Inspection logging removed for this test, focusing on instantiation
                # try:
                #     logger.info(f"Inspecting Matcher (imported): {Matcher}")
                #     logger.info(f"Inspecting type(Matcher): {type(Matcher)}")
                #     logger.info(f"Inspecting inspect.getfile(Matcher): {inspect.getfile(Matcher)}")
                #     logger.info(f"Inspecting inspect.signature(Matcher.__init__): {inspect.signature(Matcher.__init__)}")
                # except Exception as e_inspect:
                #     logger.error(f"Error during inspection: {e_inspect}")

                try:
                    # Standard Matcher instantiation
                    self.matcher = Matcher(self.nlp.vocab)
                    logger.info(f"spaCy Matcher initialized: {self.matcher}, type: {type(self.matcher)}")
                except TypeError as te:
                    logger.error(f"TypeError during Matcher instantiation: {te}", exc_info=True)
                    raise NLUProcessingError(f"TypeError creating Matcher: {te}") from te
                except Exception as e_match:
                    logger.error(f"Other error during Matcher instantiation: {e_match}", exc_info=True)
                    raise NLUProcessingError(f"Error creating Matcher: {e_match}") from e_match
            else:
                logger.error("Cannot initialize Matcher because self.nlp or self.nlp.vocab is None/invalid.")
                # This will leave self.matcher as None. _load_command_patterns will log its error and skip loading.
                # Forcing an error here if Matcher is critical for the class to function
                raise NLUProcessingError("Cannot initialize Matcher: spaCy nlp object or its vocab is invalid.")

            # Temporarily disable HF pipelines for focused Matcher debugging
            self.intent_pipeline = None
            logger.info("Intent pipeline loading (temporarily) disabled.")
            self.sentiment_pipeline = None
            logger.info("Sentiment pipeline loading (temporarily) disabled.")

            logger.info("Reached end of _load_models' try-block successfully.")

        except OSError as e: # Specifically for spacy.load()
            logger.error(f"Could not load spaCy model '{self.config.spacy_model_name}'. Error: {e}", exc_info=True)
            raise NLUProcessingError(f"Failed to load spaCy model: {self.config.spacy_model_name}") from e
        except NLUProcessingError: # Re-raise NLUProcessingErrors from Matcher init etc.
            raise
        except Exception as e_load_models: # Catch any other unexpected error during _load_models
            logger.error(f"Unexpected error during _load_models: {e_load_models}", exc_info=True)
            raise NLUProcessingError(f"Unexpected error during NLU model loading: {e_load_models}") from e_load_models
        finally:
            logger.info(f"Exiting _load_models. self.matcher is now: {self.matcher}, self.nlp is: {self.nlp}")

    def _load_command_patterns(self, patterns_file_path: str):
        logger.info(f"Entering _load_command_patterns. self.matcher is: {self.matcher}")
        if not self.matcher:
            logger.error("Matcher is None at the start of _load_command_patterns. Cannot load command patterns.")
            return
        try:
            resolved_path = Path(__file__).parent / patterns_file_path
            logger.info(f"_load_command_patterns: Attempting to load from resolved_path: {resolved_path}")
            if not resolved_path.is_file():
                logger.info(f"Resolved_path {resolved_path} not found, trying patterns_file_path as absolute: {patterns_file_path}")
                resolved_path = Path(patterns_file_path)

            logger.info(f"_load_command_patterns: Final resolved_path is: {resolved_path}, is_file: {resolved_path.is_file()}")
            if not resolved_path.is_file():
                logger.error(f"Command patterns file not found at '{patterns_file_path}' (final check at {resolved_path}).")
                return

            with open(resolved_path, 'r', encoding='utf-8') as f:
                self.command_rules = json.load(f)

            for rule_config in self.command_rules:
                command_name = rule_config["command_name"]
                patterns = rule_config["patterns"]
                self.matcher.add(command_name, patterns)
            logger.info(f"Loaded {len(self.command_rules)} command rules with {sum(len(r['patterns']) for r in self.command_rules)} patterns from '{resolved_path}'.")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from command patterns file '{resolved_path}': {e}")
            self.command_rules = []
        except Exception as e: # Catch other errors like invalid pattern format for matcher.add
            logger.error(f"Error loading or adding command patterns from '{resolved_path}': {e}", exc_info=True)
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
        llm_intent: Optional[Intent] = None
        # HF Intent pipeline temporarily disabled
        # if self.intent_pipeline:
        #     ...
        return llm_intent


    def _parse_command_from_doc(self, doc: Doc) -> Optional[ParsedCommand]:
        if not self.matcher or not self.command_rules:
            logger.debug("_parse_command_from_doc: Matcher or command_rules not available.")
            return None

        matches = self.matcher(doc)
        if not matches:
            logger.debug("_parse_command_from_doc: No matches found by Matcher.")
            return None

        logger.debug(f"_parse_command_from_doc: Found {len(matches)} raw matches.")

        # Simple strategy: use the first match. Could be more sophisticated.
        # Sort matches by start token and then by length (longest match first for overlapping patterns)
        # Spacy matches are (match_id, start_token_idx, end_token_idx)

        # Iterate through spaCy's matches, sorted to prefer longer matches if they overlap
        for match_id, start, end in sorted(matches, key=lambda m: (m[1], m[2] - m[1]), reverse=True):
            command_name_from_hash = self.nlp.vocab.strings[match_id] # Get string name from hash
            matched_span = doc[start:end]
            logger.debug(f"Processing match: ID={command_name_from_hash}, span='{matched_span.text}'")


            matched_rule_config = None
            for rule_cfg in self.command_rules:
                if rule_cfg["command_name"] == command_name_from_hash:
                    matched_rule_config = rule_cfg
                    break

            if not matched_rule_config:
                logger.warning(f"Could not find rule config for command_name '{command_name_from_hash}'. Skipping this match.")
                continue

            logger.info(f"Found command pattern match for '{matched_rule_config['command_name']}' text: '{matched_span.text}'")

            parameters: Dict[str, Any] = {}
            parameters.update(matched_rule_config.get("default_params", {}))

            # Placeholder for actual parameter extraction based on "CAPTURE" directives
            # This needs to be properly implemented by iterating through the matched_rule_config's patterns
            # and aligning them with the matched_span's tokens to extract based on "CAPTURE" keys.
            # For now, we'll log that this step is pending proper implementation.
            if any("CAPTURE" in token_pattern for pattern_list in matched_rule_config.get("patterns", []) for token_pattern in pattern_list):
                 logger.warning(f"Command '{matched_rule_config['command_name']}': Parameter extraction from 'CAPTURE' keys in patterns is not fully implemented yet. Parameters dict may be incomplete.")

            # Basic/Conceptual Parameter Extraction (Needs proper implementation)
            # This is a very naive way and won't work with current "CAPTURE" in JSON
            # It should iterate through the *specific pattern* that matched within the rule.
            # For now, let's use the entity_mapping as a guide, though it's also not fully integrated.
            for token in matched_span: # Iterate tokens in the matched span
                for pattern_group in matched_rule_config.get("patterns", []): # Iterate lists of patterns
                    for token_pattern_in_rule in pattern_group: # Iterate individual token patterns
                        if token_pattern_in_rule.get("CAPTURE") and token.lower_ == token_pattern_in_rule.get("LOWER"):
                             capture_name = token_pattern_in_rule["CAPTURE"]
                             param_name = matched_rule_config.get("entity_mapping", {}).get(capture_name, capture_name)
                             parameters[param_name] = token.text # Or matched_span[token.i-start : ...].text for multi-token captures

            # Post-processing (as before)
            post_processing_directives = matched_rule_config.get("post_processing", [])
            for directive in post_processing_directives:
                transform_type = directive.get("transform")
                params_to_process = directive.get("params", [])

                if transform_type == "TRANSFORM_STRIP" and params_to_process:
                    for p_name in params_to_process:
                        if p_name in parameters and isinstance(parameters[p_name], str):
                            parameters[p_name] = parameters[p_name].strip()
                elif transform_type == "TRANSFORM_TO_INT" and params_to_process:
                     for p_name in params_to_process:
                        if p_name in parameters:
                            try: parameters[p_name] = int(str(parameters[p_name]).strip())
                            except ValueError: logger.warning(f"Could not convert param '{p_name}' value '{parameters[p_name]}' to int.")
                elif transform_type == "TRANSFORM_LOWERCASE_AND_STRIP" and params_to_process:
                    for p_name in params_to_process:
                        if p_name in parameters and isinstance(parameters[p_name], str):
                            parameters[p_name] = parameters[p_name].lower().strip()
                elif transform_type == "TRANSFORM_PREFIX_DEFAULT_USER_ID" and params_to_process:
                    for p_name in params_to_process:
                         parameters[p_name] = self.config.default_user_identifier # Assuming default_user_identifier is in NLUConfig
                elif transform_type == "TRANSFORM_EXTRACT_KEY_VALUE_PAIRS_V2" and len(params_to_process) == 2:
                    source_param_name, target_param_name = params_to_process
                    kv_string = parameters.get(source_param_name)
                    if isinstance(kv_string, str) and kv_string.strip():
                        # Regex for "key is value" or "key:value", allowing spaces in keys and values.
                        # Non-greedy value capture, stops at next comma or end of string.
                        # Positive lookahead `(?=...)` ensures the comma is part of a new key-value pair or it's the end.
                        kv_regex = re.compile(r"([\w\s-]+?)\s*[:is](?!=\s*http)\s*([\w\s.,'@\-\/%+'\"()]+?)(?=\s*,\s*[\w\s-]+?\s*[:is](?!=\s*http)|$)", re.IGNORECASE)
                        extracted_pairs = {key.strip(): val.strip() for key, val in kv_regex.findall(kv_string)}
                        if extracted_pairs: parameters[target_param_name] = extracted_pairs
                        else: parameters[target_param_name] = {} # Return empty dict if no pairs found
                        parameters.pop(source_param_name, None) # Remove original string
                    elif kv_string is None or not kv_string.strip(): # If source is empty or None
                        parameters[target_param_name] = None # Set target to None or {} as appropriate
                        if source_param_name in parameters : parameters.pop(source_param_name, None)

            # This part for derive_status_filter needs to be re-thought with better capture mechanisms
            # For now, it's unlikely to work correctly based on current param extraction.
            # ... (original derive_status_filter logic, which is likely not working) ...

            return ParsedCommand(command=matched_rule_config["command_name"], parameters=parameters)
        return None


    def _analyze_sentiment(self, text: str) -> Optional[Sentiment]:
        if not self.sentiment_pipeline:
            return None
        # ... (original sentiment logic) ...
        return None # Temporarily disabled

    def process_text(self, text: str, context: Optional[dict] = None) -> NLUResult:
        if not self.nlp:
            raise NLUProcessingError("NLUProcessor is not initialized properly (spaCy model missing).")
        start_time = time.time()
        doc = self.nlp(text)
        spacy_entities = self._extract_spacy_entities(doc)

        # Intent Recognition is temporarily disabled for Matcher debugging
        detected_intent: Optional[Intent] = None
        # detected_intent = self._recognize_intent(text, doc=doc)

        detected_sentiment = self._analyze_sentiment(text) # Also temporarily disabled

        parsed_command = None
        if self.config.enable_rule_based_entities:
            parsed_command = self._parse_command_from_doc(doc)
            if parsed_command:
                logger.info(f"Parsed command: {parsed_command.command} with params: {parsed_command.parameters}")
                for rule_cfg in self.command_rules:
                    if rule_cfg["command_name"] == parsed_command.command:
                        intent_aliases = rule_cfg.get("intent_aliases", [])
                        if intent_aliases:
                            # If command is parsed, use its first alias as intent (high confidence)
                            # This overrides the (currently disabled) LLM intent
                            new_intent_name = intent_aliases[0]
                            logger.info(f"Command parsing via Matcher identified command '{parsed_command.command}', setting intent to '{new_intent_name}'.")
                            detected_intent = Intent(name=new_intent_name, confidence=0.98) # High confidence for rule match
                        break

        # Fallback if no command-driven intent and no LLM intent (currently disabled)
        if not detected_intent and text and text.strip():
             # Basic fallback: if intent pipeline was disabled or failed, and no command matched,
             # we might assign a generic intent or leave it None.
             # For tests, if command parsing fails, detected_intent will be None.
             pass


        tokens = [token.text for token in doc]
        sentences = [sent.text for sent in doc.sents]
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        return NLUResult(
            raw_text=text,
            detected_intent=detected_intent,
            entities=spacy_entities,
            sentiment=detected_sentiment,
            parsed_command=parsed_command,
            language=doc.lang_ if doc.lang_ else self.config.default_language,
            processing_time_ms=processing_time_ms,
            tokens=tokens,
            sentences=sentences
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Initializing NLUProcessor with default config...")
    try:
        # Ensure en_core_web_sm is downloaded: python -m spacy download en_core_web_sm
        test_config = NLUConfig(spacy_model_name="en_core_web_sm", command_patterns_file="nlu_command_patterns.json")
        processor = NLUProcessor(config=test_config)

        logger.info("\n--- Testing Objective Management Commands ---")
        objective_test_cases = [
            "add objective: conquer the world priority 1",
            "my new goal is to learn spaCy",
            "list my active objectives",
            "show all objectives",
            "list completed goals for project AlphaBeta",
            "set objective userobj_1234-abcd-5678-efgh status to completed",
            "mark goal userobj_aaaa-bbbb-cccc-dddd as on-hold"
        ]

        for i, test_case in enumerate(objective_test_cases):
            logger.info(f"Processing objective command test case {i+1}: \"{test_case}\"")
            result = processor.process_text(test_case)
            print(f"  Raw Result: {result}")
            if result.parsed_command:
                print(f"  Command: {result.parsed_command.command}")
                print(f"  Parameters: {result.parsed_command.parameters}")
            else:
                print("  No command parsed.")
            if result.entities:
                print(f"  Entities: {[e.to_dict() for e in result.entities]}")
            print("-" * 20)

    except NLUProcessingError as e:
        logger.error(f"NLU Processing Error during example usage: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during example usage: {e}", exc_info=True)

"""
TODO:
- Full implementation of parameter extraction in `_parse_command_from_doc` using spaCy Matcher's `on_match` callback or by carefully designing patterns and then extracting token attributes from the matched span based on the known structure of the pattern that fired. The current "CAPTURE" key in JSON is not standard spaCy and needs a proper handling mechanism.
- Re-enable EntityRuler and HuggingFace pipelines once Matcher/command parsing is stable.
- Refine intent alias logic to better integrate rule-based command detection with LLM-based intent recognition.
"""
