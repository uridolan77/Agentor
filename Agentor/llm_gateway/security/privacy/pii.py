"""
PII detection and redaction for the LLM Gateway.

This module provides detection and redaction of personally identifiable information (PII),
including pattern-based detection, entity recognition, and configurable redaction.
"""

import re
import logging
import hashlib
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Pattern, Callable
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    DOB = "dob"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    CUSTOM = "custom"


class PIIDetection:
    """Detection result for PII."""
    
    def __init__(
        self,
        pii_type: PIIType,
        value: str,
        confidence: float,
        start_pos: int,
        end_pos: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PII detection result.
        
        Args:
            pii_type: Type of PII detected
            value: Detected PII value
            confidence: Confidence score (0.0 to 1.0)
            start_pos: Start position in text
            end_pos: End position in text
            metadata: Additional metadata
        """
        self.pii_type = pii_type
        self.value = value
        self.confidence = confidence
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection result to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "pii_type": self.pii_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation
        """
        return f"PII detected: {self.pii_type.value}, Confidence: {self.confidence:.2f}, Position: {self.start_pos}-{self.end_pos}"


class RedactionMethod(Enum):
    """Methods for redacting PII."""
    MASK = "mask"  # Replace with a mask character (e.g., *)
    REPLACE = "replace"  # Replace with a placeholder (e.g., [EMAIL])
    HASH = "hash"  # Replace with a hash of the value
    PSEUDONYMIZE = "pseudonymize"  # Replace with a consistent pseudonym
    TOKENIZE = "tokenize"  # Replace with a token that can be reversed
    CUSTOM = "custom"  # Custom redaction method


class RedactionConfig:
    """Configuration for PII redaction."""
    
    def __init__(
        self,
        method: RedactionMethod = RedactionMethod.REPLACE,
        mask_char: str = "*",
        preserve_length: bool = False,
        placeholder_format: str = "[{pii_type}]",
        hash_algorithm: str = "sha256",
        hash_length: int = 8,
        pseudonym_map: Optional[Dict[str, Dict[str, str]]] = None,
        token_map: Optional[Dict[str, Dict[str, str]]] = None,
        custom_redactor: Optional[Callable[[str, PIIType], str]] = None
    ):
        """
        Initialize redaction configuration.
        
        Args:
            method: Redaction method
            mask_char: Character to use for masking
            preserve_length: Whether to preserve the length of the original value
            placeholder_format: Format string for placeholder replacement
            hash_algorithm: Hash algorithm to use
            hash_length: Length of hash to use
            pseudonym_map: Map of PII values to pseudonyms
            token_map: Map of PII values to tokens
            custom_redactor: Custom redaction function
        """
        self.method = method
        self.mask_char = mask_char
        self.preserve_length = preserve_length
        self.placeholder_format = placeholder_format
        self.hash_algorithm = hash_algorithm
        self.hash_length = hash_length
        self.pseudonym_map = pseudonym_map or {}
        self.token_map = token_map or {}
        self.custom_redactor = custom_redactor
    
    def get_redacted_value(self, value: str, pii_type: PIIType) -> str:
        """
        Get redacted value for PII.
        
        Args:
            value: Original PII value
            pii_type: Type of PII
            
        Returns:
            Redacted value
        """
        if self.method == RedactionMethod.MASK:
            if self.preserve_length:
                return self.mask_char * len(value)
            else:
                return self.mask_char * min(len(value), 8)
        
        elif self.method == RedactionMethod.REPLACE:
            return self.placeholder_format.format(pii_type=pii_type.value.upper())
        
        elif self.method == RedactionMethod.HASH:
            # Hash the value
            hash_obj = hashlib.new(self.hash_algorithm)
            hash_obj.update(value.encode())
            hash_value = hash_obj.hexdigest()[:self.hash_length]
            
            return f"[{pii_type.value.upper()}:{hash_value}]"
        
        elif self.method == RedactionMethod.PSEUDONYMIZE:
            # Get or create pseudonym
            pii_type_str = pii_type.value
            if pii_type_str not in self.pseudonym_map:
                self.pseudonym_map[pii_type_str] = {}
            
            if value not in self.pseudonym_map[pii_type_str]:
                # Create pseudonym based on PII type
                if pii_type == PIIType.EMAIL:
                    username = f"user{len(self.pseudonym_map[pii_type_str]) + 1}"
                    domain = "example.com"
                    pseudonym = f"{username}@{domain}"
                elif pii_type == PIIType.PHONE:
                    pseudonym = f"555-{len(self.pseudonym_map[pii_type_str]) + 1:04d}"
                elif pii_type == PIIType.NAME:
                    pseudonym = f"Person{len(self.pseudonym_map[pii_type_str]) + 1}"
                else:
                    # Generic pseudonym
                    pseudonym = f"{pii_type.value.upper()}{len(self.pseudonym_map[pii_type_str]) + 1}"
                
                self.pseudonym_map[pii_type_str][value] = pseudonym
            
            return self.pseudonym_map[pii_type_str][value]
        
        elif self.method == RedactionMethod.TOKENIZE:
            # Get or create token
            pii_type_str = pii_type.value
            if pii_type_str not in self.token_map:
                self.token_map[pii_type_str] = {}
            
            if value not in self.token_map[pii_type_str]:
                # Create token
                token = str(uuid.uuid4())
                self.token_map[pii_type_str][value] = token
            
            return f"[TOKEN:{self.token_map[pii_type_str][value]}]"
        
        elif self.method == RedactionMethod.CUSTOM and self.custom_redactor:
            # Use custom redactor
            return self.custom_redactor(value, pii_type)
        
        # Default: return placeholder
        return self.placeholder_format.format(pii_type=pii_type.value.upper())
    
    def get_original_value(self, redacted_value: str, pii_type: PIIType) -> Optional[str]:
        """
        Get original value from redacted value (for tokenization).
        
        Args:
            redacted_value: Redacted value
            pii_type: Type of PII
            
        Returns:
            Original value or None if not found
        """
        if self.method != RedactionMethod.TOKENIZE:
            return None
        
        # Extract token
        token_match = re.match(r"\[TOKEN:([a-f0-9-]+)\]", redacted_value)
        if not token_match:
            return None
        
        token = token_match.group(1)
        
        # Look up original value
        pii_type_str = pii_type.value
        if pii_type_str not in self.token_map:
            return None
        
        # Find original value by token
        for original, stored_token in self.token_map[pii_type_str].items():
            if stored_token == token:
                return original
        
        return None


class PatternBasedPIIDetector:
    """Pattern-based detector for PII."""
    
    # Default patterns for different PII types
    DEFAULT_PATTERNS = {
        PIIType.EMAIL: [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        ],
        PIIType.PHONE: [
            r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ],
        PIIType.SSN: [
            r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"
        ],
        PIIType.CREDIT_CARD: [
            r"\b(?:\d{4}[- ]?){3}\d{4}\b",
            r"\b\d{13,16}\b"
        ],
        PIIType.IP_ADDRESS: [
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            r"\b([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}\b"
        ],
        PIIType.ADDRESS: [
            r"\b\d+\s+[a-zA-Z0-9\s,]+\b(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\b",
            r"\b\d{5}(?:[-\s]\d{4})?\b"  # ZIP code
        ],
        PIIType.DOB: [
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"
        ],
        PIIType.PASSPORT: [
            r"\b[A-Z]{1,2}\d{6,9}\b"
        ],
        PIIType.DRIVER_LICENSE: [
            r"\b[A-Z]\d{7}\b",
            r"\b[A-Z]{1,2}\d{5,7}\b"
        ],
        PIIType.BANK_ACCOUNT: [
            r"\b\d{8,17}\b"
        ],
        PIIType.API_KEY: [
            r"\b(?:api[_-]?key|token)[=:]\s*['\"]([\w\-]{20,})['\"]\b",
            r"\b[A-Za-z0-9_-]{32,}\b"
        ]
    }
    
    def __init__(
        self,
        patterns: Optional[Dict[PIIType, List[str]]] = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        min_confidence: float = 0.7,
        enable_all: bool = True,
        enabled_types: Optional[Set[PIIType]] = None
    ):
        """
        Initialize pattern-based detector.
        
        Args:
            patterns: Dictionary mapping PII types to lists of regex patterns
            custom_patterns: Dictionary mapping custom PII type names to lists of regex patterns
            min_confidence: Minimum confidence threshold
            enable_all: Whether to enable all PII types
            enabled_types: Set of enabled PII types (if enable_all is False)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self.min_confidence = min_confidence
        self.enable_all = enable_all
        self.enabled_types = enabled_types or set()
        
        # Add custom patterns
        if custom_patterns:
            for custom_type, pattern_list in custom_patterns.items():
                custom_pii_type = PIIType.CUSTOM
                if custom_type not in self.patterns:
                    self.patterns[custom_pii_type] = []
                self.patterns[custom_pii_type].extend(pattern_list)
        
        # Compile patterns
        self.compiled_patterns = {}
        for pii_type, pattern_list in self.patterns.items():
            self.compiled_patterns[pii_type] = [re.compile(pattern) for pattern in pattern_list]
    
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detection results
        """
        results = []
        
        # Check each PII type
        for pii_type, compiled_patterns in self.compiled_patterns.items():
            # Skip if not enabled
            if not self.enable_all and pii_type not in self.enabled_types:
                continue
            
            # Check each pattern
            for compiled_pattern in compiled_patterns:
                # Find all matches
                for match in compiled_pattern.finditer(text):
                    # Calculate confidence based on match length and pattern complexity
                    match_length = len(match.group(0))
                    pattern_complexity = len(compiled_pattern.pattern)
                    confidence = min(0.5 + (match_length / 100) + (pattern_complexity / 200), 1.0)
                    
                    # Skip if confidence is below threshold
                    if confidence < self.min_confidence:
                        continue
                    
                    # Create detection result
                    detection = PIIDetection(
                        pii_type=pii_type,
                        value=match.group(0),
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        metadata={
                            "pattern": compiled_pattern.pattern,
                            "groups": match.groups(),
                            "named_groups": match.groupdict()
                        }
                    )
                    
                    results.append(detection)
        
        return results


class NamedEntityPIIDetector:
    """Named entity recognition based detector for PII."""
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        enable_spacy: bool = False,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize named entity detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            enable_spacy: Whether to enable spaCy for NER
            spacy_model: spaCy model to use
        """
        self.min_confidence = min_confidence
        self.enable_spacy = enable_spacy
        self.spacy_model = spacy_model
        self.nlp = None
        
        # Initialize spaCy if enabled
        if self.enable_spacy:
            try:
                import spacy
                self.nlp = spacy.load(self.spacy_model)
                logger.info(f"Initialized spaCy model {self.spacy_model} for NER-based PII detection")
            except ImportError:
                logger.warning("spaCy not installed, NER-based PII detection disabled")
                self.enable_spacy = False
            except Exception as e:
                logger.warning(f"Failed to load spaCy model {self.spacy_model}: {e}")
                self.enable_spacy = False
    
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in text using named entity recognition.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detection results
        """
        results = []
        
        # Skip if spaCy is not enabled or failed to load
        if not self.enable_spacy or not self.nlp:
            return results
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Map spaCy entity types to PII types
        entity_map = {
            "PERSON": PIIType.NAME,
            "ORG": PIIType.NAME,  # Organizations can be PII in some contexts
            "GPE": PIIType.ADDRESS,  # Geopolitical entities (cities, countries)
            "LOC": PIIType.ADDRESS,  # Non-GPE locations
            "DATE": PIIType.DOB,  # Dates can be PII if they're birth dates
            "CARDINAL": None,  # Numbers (may be PII depending on context)
            "MONEY": None,  # Monetary values
            "PRODUCT": None,  # Products
            "EVENT": None,  # Events
            "WORK_OF_ART": None,  # Titles of books, songs, etc.
            "LAW": None,  # Laws, bills, etc.
            "LANGUAGE": None,  # Languages
            "FAC": PIIType.ADDRESS,  # Facilities (buildings, airports, highways)
            "NORP": None,  # Nationalities, religious or political groups
        }
        
        # Process entities
        for ent in doc.ents:
            pii_type = entity_map.get(ent.label_)
            if pii_type:
                # Calculate confidence based on entity label and length
                confidence = 0.7  # Base confidence for NER
                if len(ent.text) > 3:
                    confidence += 0.1  # Longer entities are more likely to be PII
                
                # Skip if confidence is below threshold
                if confidence < self.min_confidence:
                    continue
                
                # Create detection result
                detection = PIIDetection(
                    pii_type=pii_type,
                    value=ent.text,
                    confidence=confidence,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    metadata={
                        "entity_type": ent.label_,
                        "method": "ner"
                    }
                )
                
                results.append(detection)
        
        return results


class PIIDetector:
    """Comprehensive detector for PII."""
    
    def __init__(
        self,
        pattern_detector: Optional[PatternBasedPIIDetector] = None,
        ner_detector: Optional[NamedEntityPIIDetector] = None,
        min_confidence: float = 0.7,
        enable_pattern: bool = True,
        enable_ner: bool = False
    ):
        """
        Initialize PII detector.
        
        Args:
            pattern_detector: Pattern-based detector
            ner_detector: Named entity detector
            min_confidence: Minimum confidence threshold
            enable_pattern: Whether to enable pattern-based detection
            enable_ner: Whether to enable NER-based detection
        """
        self.pattern_detector = pattern_detector or PatternBasedPIIDetector()
        self.ner_detector = ner_detector or NamedEntityPIIDetector()
        self.min_confidence = min_confidence
        self.enable_pattern = enable_pattern
        self.enable_ner = enable_ner
    
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of PII detection results
        """
        results = []
        
        # Pattern-based detection
        if self.enable_pattern:
            pattern_results = self.pattern_detector.detect(text)
            results.extend(pattern_results)
        
        # NER-based detection
        if self.enable_ner:
            ner_results = self.ner_detector.detect(text)
            results.extend(ner_results)
        
        # Filter by confidence
        results = [result for result in results if result.confidence >= self.min_confidence]
        
        # Sort by position
        results.sort(key=lambda x: x.start_pos)
        
        # Remove overlapping detections (keep higher confidence)
        non_overlapping = []
        for i, detection in enumerate(results):
            # Check if this detection overlaps with any previous non-overlapping detection
            overlaps = False
            for prev in non_overlapping:
                if (detection.start_pos < prev.end_pos and detection.end_pos > prev.start_pos):
                    overlaps = True
                    # If current detection has higher confidence, replace previous
                    if detection.confidence > prev.confidence:
                        non_overlapping.remove(prev)
                        non_overlapping.append(detection)
                    break
            
            if not overlaps:
                non_overlapping.append(detection)
        
        # Sort by position again
        non_overlapping.sort(key=lambda x: x.start_pos)
        
        return non_overlapping


class PIIRedactor:
    """Redactor for PII in text."""
    
    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        config: Optional[Dict[PIIType, RedactionConfig]] = None,
        default_config: Optional[RedactionConfig] = None
    ):
        """
        Initialize PII redactor.
        
        Args:
            detector: PII detector
            config: Dictionary mapping PII types to redaction configurations
            default_config: Default redaction configuration
        """
        self.detector = detector or PIIDetector()
        self.config = config or {}
        self.default_config = default_config or RedactionConfig()
    
    def redact(self, text: str) -> Tuple[str, List[PIIDetection]]:
        """
        Redact PII in text.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted text, list of detections)
        """
        # Detect PII
        detections = self.detector.detect(text)
        
        # If no detections, return original text
        if not detections:
            return text, []
        
        # Redact PII
        redacted_text = text
        offset = 0
        
        for detection in detections:
            # Get redaction configuration for this PII type
            config = self.config.get(detection.pii_type, self.default_config)
            
            # Get redacted value
            redacted_value = config.get_redacted_value(detection.value, detection.pii_type)
            
            # Replace in text
            start_pos = detection.start_pos + offset
            end_pos = detection.end_pos + offset
            
            redacted_text = redacted_text[:start_pos] + redacted_value + redacted_text[end_pos:]
            
            # Update offset
            offset += len(redacted_value) - (end_pos - start_pos)
        
        return redacted_text, detections
    
    def redact_json(self, json_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[PIIDetection]]:
        """
        Redact PII in JSON data.
        
        Args:
            json_data: JSON data to redact
            
        Returns:
            Tuple of (redacted JSON data, list of detections)
        """
        all_detections = []
        
        def redact_value(value):
            nonlocal all_detections
            
            if isinstance(value, str):
                redacted_value, detections = self.redact(value)
                all_detections.extend(detections)
                return redacted_value
            elif isinstance(value, dict):
                return {k: redact_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [redact_value(item) for item in value]
            else:
                return value
        
        redacted_data = redact_value(json_data)
        
        return redacted_data, all_detections
    
    def restore(self, redacted_text: str, detections: List[PIIDetection]) -> str:
        """
        Restore original text from redacted text (for tokenization).
        
        Args:
            redacted_text: Redacted text
            detections: List of PII detections
            
        Returns:
            Restored text
        """
        # Only works for tokenization
        restored_text = redacted_text
        
        # Process detections in reverse order to avoid position issues
        for detection in reversed(detections):
            # Get redaction configuration for this PII type
            config = self.config.get(detection.pii_type, self.default_config)
            
            # Skip if not tokenization
            if config.method != RedactionMethod.TOKENIZE:
                continue
            
            # Find token in redacted text
            token_pattern = r"\[TOKEN:([a-f0-9-]+)\]"
            for match in re.finditer(token_pattern, restored_text):
                token = match.group(1)
                
                # Check if this token corresponds to this detection
                original_value = None
                for original, stored_token in config.token_map.get(detection.pii_type.value, {}).items():
                    if stored_token == token:
                        original_value = original
                        break
                
                if original_value:
                    # Replace token with original value
                    restored_text = restored_text[:match.start()] + original_value + restored_text[match.end():]
        
        return restored_text
