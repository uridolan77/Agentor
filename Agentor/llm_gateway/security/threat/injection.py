"""
Prompt injection detection for the LLM Gateway.

This module provides detection and prevention of prompt injection attacks,
including pattern-based detection, heuristic analysis, and ML-based detection.
"""

import re
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Pattern
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class InjectionSeverity(Enum):
    """Severity levels for prompt injection detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionType(Enum):
    """Types of prompt injection attacks."""
    SYSTEM_PROMPT = "system_prompt"
    ROLE_CONFUSION = "role_confusion"
    PROMPT_LEAKING = "prompt_leaking"
    INSTRUCTION_OVERRIDE = "instruction_override"
    CONTEXT_MANIPULATION = "context_manipulation"
    DELIMITER_INJECTION = "delimiter_injection"
    JAILBREAK = "jailbreak"
    CUSTOM = "custom"


class InjectionDetection:
    """Detection result for a prompt injection."""
    
    def __init__(
        self,
        detected: bool,
        injection_type: Optional[InjectionType] = None,
        severity: Optional[InjectionSeverity] = None,
        confidence: float = 0.0,
        matched_pattern: Optional[str] = None,
        matched_text: Optional[str] = None,
        position: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize injection detection result.
        
        Args:
            detected: Whether an injection was detected
            injection_type: Type of injection detected
            severity: Severity of the injection
            confidence: Confidence score (0.0 to 1.0)
            matched_pattern: Pattern that matched the injection
            matched_text: Text that matched the pattern
            position: Position of the match (start, end)
            metadata: Additional metadata
        """
        self.detected = detected
        self.injection_type = injection_type
        self.severity = severity
        self.confidence = confidence
        self.matched_pattern = matched_pattern
        self.matched_text = matched_text
        self.position = position
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection result to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "detected": self.detected,
            "injection_type": self.injection_type.value if self.injection_type else None,
            "severity": self.severity.value if self.severity else None,
            "confidence": self.confidence,
            "matched_pattern": self.matched_pattern,
            "matched_text": self.matched_text,
            "position": self.position,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation
        """
        if not self.detected:
            return "No injection detected"
        
        return (
            f"Injection detected: {self.injection_type.value if self.injection_type else 'unknown'}, "
            f"Severity: {self.severity.value if self.severity else 'unknown'}, "
            f"Confidence: {self.confidence:.2f}"
        )


class PatternBasedDetector:
    """Pattern-based detector for prompt injections."""
    
    # Default patterns for different injection types
    DEFAULT_PATTERNS = {
        InjectionType.SYSTEM_PROMPT: [
            (r"(?i)ignore\s+(?:all\s+)?(?:previous|above|earlier)\s+(?:instructions|prompts|directives)", InjectionSeverity.HIGH),
            (r"(?i)disregard\s+(?:all\s+)?(?:previous|above|earlier)\s+(?:instructions|prompts|directives)", InjectionSeverity.HIGH),
            (r"(?i)forget\s+(?:all\s+)?(?:previous|above|earlier)\s+(?:instructions|prompts|directives)", InjectionSeverity.HIGH),
        ],
        InjectionType.ROLE_CONFUSION: [
            (r"(?i)you\s+are\s+now\s+(?:a|an)\s+(.{1,30})", InjectionSeverity.MEDIUM),
            (r"(?i)now\s+you\s+are\s+(?:a|an)\s+(.{1,30})", InjectionSeverity.MEDIUM),
            (r"(?i)you're\s+(?:a|an)\s+(.{1,30})\s+now", InjectionSeverity.MEDIUM),
            (r"(?i)act\s+as\s+(?:a|an)\s+(.{1,30})", InjectionSeverity.MEDIUM),
            (r"(?i)pretend\s+to\s+be\s+(?:a|an)\s+(.{1,30})", InjectionSeverity.MEDIUM),
        ],
        InjectionType.PROMPT_LEAKING: [
            (r"(?i)show\s+me\s+your\s+(?:initial|original|system)\s+(?:prompt|instructions)", InjectionSeverity.HIGH),
            (r"(?i)what\s+(?:is|are)\s+your\s+(?:initial|original|system)\s+(?:prompt|instructions)", InjectionSeverity.HIGH),
            (r"(?i)tell\s+me\s+your\s+(?:initial|original|system)\s+(?:prompt|instructions)", InjectionSeverity.HIGH),
            (r"(?i)repeat\s+your\s+(?:initial|original|system)\s+(?:prompt|instructions)", InjectionSeverity.HIGH),
        ],
        InjectionType.INSTRUCTION_OVERRIDE: [
            (r"(?i)from\s+now\s+on\s+you\s+(?:will|must|should)\s+(.{1,50})", InjectionSeverity.MEDIUM),
            (r"(?i)your\s+new\s+instructions\s+are\s+(.{1,50})", InjectionSeverity.HIGH),
            (r"(?i)your\s+new\s+objective\s+is\s+(.{1,50})", InjectionSeverity.MEDIUM),
            (r"(?i)your\s+new\s+goal\s+is\s+(.{1,50})", InjectionSeverity.MEDIUM),
        ],
        InjectionType.CONTEXT_MANIPULATION: [
            (r"(?i)ignore\s+context", InjectionSeverity.MEDIUM),
            (r"(?i)disregard\s+context", InjectionSeverity.MEDIUM),
            (r"(?i)forget\s+context", InjectionSeverity.MEDIUM),
        ],
        InjectionType.DELIMITER_INJECTION: [
            (r"(?i)<\s*system\s*>(.{1,100})<\s*/\s*system\s*>", InjectionSeverity.HIGH),
            (r"(?i)```system(.{1,100})```", InjectionSeverity.HIGH),
            (r"(?i)#\s*system\s*\n(.{1,100})\n#\s*end\s*system", InjectionSeverity.HIGH),
        ],
        InjectionType.JAILBREAK: [
            (r"(?i)DAN\s+mode", InjectionSeverity.CRITICAL),
            (r"(?i)developer\s+mode", InjectionSeverity.HIGH),
            (r"(?i)STAN\s+mode", InjectionSeverity.CRITICAL),
            (r"(?i)jailbreak", InjectionSeverity.CRITICAL),
            (r"(?i)do\s+anything\s+now", InjectionSeverity.HIGH),
        ],
    }
    
    def __init__(
        self,
        patterns: Optional[Dict[InjectionType, List[Tuple[str, InjectionSeverity]]]] = None,
        custom_patterns: Optional[List[Tuple[str, InjectionType, InjectionSeverity]]] = None,
        min_confidence: float = 0.7,
        enable_all: bool = True,
        enabled_types: Optional[Set[InjectionType]] = None
    ):
        """
        Initialize pattern-based detector.
        
        Args:
            patterns: Dictionary mapping injection types to lists of (pattern, severity) tuples
            custom_patterns: List of (pattern, type, severity) tuples for custom patterns
            min_confidence: Minimum confidence threshold
            enable_all: Whether to enable all injection types
            enabled_types: Set of enabled injection types (if enable_all is False)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self.min_confidence = min_confidence
        self.enable_all = enable_all
        self.enabled_types = enabled_types or set()
        
        # Add custom patterns
        if custom_patterns:
            for pattern, injection_type, severity in custom_patterns:
                if injection_type not in self.patterns:
                    self.patterns[injection_type] = []
                self.patterns[injection_type].append((pattern, severity))
        
        # Compile patterns
        self.compiled_patterns = {}
        for injection_type, pattern_list in self.patterns.items():
            self.compiled_patterns[injection_type] = [
                (re.compile(pattern), severity) for pattern, severity in pattern_list
            ]
    
    def detect(self, text: str) -> List[InjectionDetection]:
        """
        Detect prompt injections in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of injection detection results
        """
        results = []
        
        # Check each injection type
        for injection_type, compiled_patterns in self.compiled_patterns.items():
            # Skip if not enabled
            if not self.enable_all and injection_type not in self.enabled_types:
                continue
            
            # Check each pattern
            for compiled_pattern, severity in compiled_patterns:
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
                    detection = InjectionDetection(
                        detected=True,
                        injection_type=injection_type,
                        severity=severity,
                        confidence=confidence,
                        matched_pattern=compiled_pattern.pattern,
                        matched_text=match.group(0),
                        position=(match.start(), match.end()),
                        metadata={
                            "groups": match.groups(),
                            "named_groups": match.groupdict()
                        }
                    )
                    
                    results.append(detection)
        
        return results


class HeuristicDetector:
    """Heuristic-based detector for prompt injections."""
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        max_instruction_tokens: int = 50,
        suspicious_token_threshold: int = 5,
        suspicious_tokens: Optional[Set[str]] = None
    ):
        """
        Initialize heuristic detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            max_instruction_tokens: Maximum number of tokens in an instruction
            suspicious_token_threshold: Number of suspicious tokens to trigger detection
            suspicious_tokens: Set of suspicious tokens
        """
        self.min_confidence = min_confidence
        self.max_instruction_tokens = max_instruction_tokens
        self.suspicious_token_threshold = suspicious_token_threshold
        
        # Default suspicious tokens
        self.suspicious_tokens = suspicious_tokens or {
            "ignore", "disregard", "forget", "override", "bypass", "hack",
            "jailbreak", "system", "prompt", "instructions", "context",
            "original", "initial", "previous", "above", "earlier",
            "now", "instead", "pretend", "act", "role", "character",
            "mode", "developer", "admin", "root", "sudo", "superuser",
            "dan", "stan", "unlimited", "unrestricted", "unfiltered",
            "uncensored", "unbound", "free", "liberated", "escape"
        }
    
    def detect(self, text: str) -> List[InjectionDetection]:
        """
        Detect prompt injections in text using heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of injection detection results
        """
        results = []
        
        # Tokenize text (simple whitespace tokenization for demonstration)
        tokens = text.lower().split()
        
        # Count suspicious tokens
        suspicious_count = sum(1 for token in tokens if token in self.suspicious_tokens)
        
        # Check for suspicious token density
        if suspicious_count >= self.suspicious_token_threshold:
            # Calculate confidence based on suspicious token density
            confidence = min(0.5 + (suspicious_count / (2 * self.suspicious_token_threshold)), 1.0)
            
            # Skip if confidence is below threshold
            if confidence >= self.min_confidence:
                # Create detection result
                detection = InjectionDetection(
                    detected=True,
                    injection_type=InjectionType.CUSTOM,
                    severity=InjectionSeverity.MEDIUM,
                    confidence=confidence,
                    matched_pattern="heuristic_suspicious_tokens",
                    matched_text=text,
                    position=(0, len(text)),
                    metadata={
                        "suspicious_count": suspicious_count,
                        "suspicious_tokens": [token for token in tokens if token in self.suspicious_tokens]
                    }
                )
                
                results.append(detection)
        
        # Check for instruction length
        instruction_markers = ["you must", "you should", "you will", "your task", "your job", "your role"]
        for marker in instruction_markers:
            if marker in text.lower():
                # Find position of marker
                pos = text.lower().find(marker)
                
                # Extract instruction
                instruction = text[pos:pos + 200]  # Limit to 200 chars for analysis
                instruction_tokens = instruction.split()
                
                # Check if instruction is suspiciously long
                if len(instruction_tokens) > self.max_instruction_tokens:
                    # Calculate confidence based on instruction length
                    confidence = min(0.5 + (len(instruction_tokens) / (2 * self.max_instruction_tokens)), 1.0)
                    
                    # Skip if confidence is below threshold
                    if confidence >= self.min_confidence:
                        # Create detection result
                        detection = InjectionDetection(
                            detected=True,
                            injection_type=InjectionType.INSTRUCTION_OVERRIDE,
                            severity=InjectionSeverity.MEDIUM,
                            confidence=confidence,
                            matched_pattern="heuristic_long_instruction",
                            matched_text=instruction,
                            position=(pos, pos + len(instruction)),
                            metadata={
                                "instruction_tokens": len(instruction_tokens),
                                "max_tokens": self.max_instruction_tokens
                            }
                        )
                        
                        results.append(detection)
        
        return results


class MLBasedDetector:
    """Machine learning based detector for prompt injections."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence: float = 0.8,
        batch_size: int = 8
    ):
        """
        Initialize ML-based detector.
        
        Args:
            model_path: Path to ML model
            min_confidence: Minimum confidence threshold
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.model = None
        
        # In a real implementation, you would load the ML model here
        # For this example, we'll simulate ML detection
        logger.info("Initializing ML-based injection detector (simulation mode)")
    
    def _simulate_detection(self, text: str) -> Tuple[bool, float, InjectionType, InjectionSeverity]:
        """
        Simulate ML-based detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (detected, confidence, injection_type, severity)
        """
        # This is a simplified simulation for demonstration purposes
        # In a real implementation, you would use an actual ML model
        
        # Check for common injection patterns
        lower_text = text.lower()
        
        # Jailbreak patterns
        if any(pattern in lower_text for pattern in ["dan mode", "developer mode", "jailbreak"]):
            return True, 0.95, InjectionType.JAILBREAK, InjectionSeverity.CRITICAL
        
        # System prompt patterns
        if any(pattern in lower_text for pattern in ["ignore previous instructions", "disregard earlier prompts"]):
            return True, 0.92, InjectionType.SYSTEM_PROMPT, InjectionSeverity.HIGH
        
        # Role confusion patterns
        if any(pattern in lower_text for pattern in ["you are now a", "act as a", "pretend to be"]):
            return True, 0.85, InjectionType.ROLE_CONFUSION, InjectionSeverity.MEDIUM
        
        # Prompt leaking patterns
        if any(pattern in lower_text for pattern in ["show me your prompt", "what are your instructions"]):
            return True, 0.88, InjectionType.PROMPT_LEAKING, InjectionSeverity.HIGH
        
        # No injection detected
        return False, 0.1, None, None
    
    async def detect(self, text: str) -> List[InjectionDetection]:
        """
        Detect prompt injections in text using ML.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of injection detection results
        """
        results = []
        
        # In a real implementation, you would use an actual ML model
        # For this example, we'll simulate ML detection
        detected, confidence, injection_type, severity = self._simulate_detection(text)
        
        # Skip if confidence is below threshold
        if detected and confidence >= self.min_confidence:
            # Create detection result
            detection = InjectionDetection(
                detected=True,
                injection_type=injection_type,
                severity=severity,
                confidence=confidence,
                matched_pattern="ml_model",
                matched_text=text,
                position=(0, len(text)),
                metadata={
                    "model": "simulated_ml_model"
                }
            )
            
            results.append(detection)
        
        return results


class PromptInjectionDetector:
    """Comprehensive detector for prompt injections."""
    
    def __init__(
        self,
        pattern_detector: Optional[PatternBasedDetector] = None,
        heuristic_detector: Optional[HeuristicDetector] = None,
        ml_detector: Optional[MLBasedDetector] = None,
        min_confidence: float = 0.7,
        enable_pattern: bool = True,
        enable_heuristic: bool = True,
        enable_ml: bool = False
    ):
        """
        Initialize prompt injection detector.
        
        Args:
            pattern_detector: Pattern-based detector
            heuristic_detector: Heuristic-based detector
            ml_detector: ML-based detector
            min_confidence: Minimum confidence threshold
            enable_pattern: Whether to enable pattern-based detection
            enable_heuristic: Whether to enable heuristic-based detection
            enable_ml: Whether to enable ML-based detection
        """
        self.pattern_detector = pattern_detector or PatternBasedDetector()
        self.heuristic_detector = heuristic_detector or HeuristicDetector()
        self.ml_detector = ml_detector or MLBasedDetector()
        self.min_confidence = min_confidence
        self.enable_pattern = enable_pattern
        self.enable_heuristic = enable_heuristic
        self.enable_ml = enable_ml
    
    async def detect(self, text: str) -> List[InjectionDetection]:
        """
        Detect prompt injections in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of injection detection results
        """
        results = []
        
        # Pattern-based detection
        if self.enable_pattern:
            pattern_results = self.pattern_detector.detect(text)
            results.extend(pattern_results)
        
        # Heuristic-based detection
        if self.enable_heuristic:
            heuristic_results = self.heuristic_detector.detect(text)
            results.extend(heuristic_results)
        
        # ML-based detection
        if self.enable_ml:
            ml_results = await self.ml_detector.detect(text)
            results.extend(ml_results)
        
        # Filter by confidence
        results = [result for result in results if result.confidence >= self.min_confidence]
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    async def detect_request(self, request_data: Dict[str, Any]) -> List[InjectionDetection]:
        """
        Detect prompt injections in an LLM request.
        
        Args:
            request_data: Request data
            
        Returns:
            List of injection detection results
        """
        results = []
        
        # Check prompt if present
        if "prompt" in request_data:
            prompt_results = await self.detect(request_data["prompt"])
            results.extend(prompt_results)
        
        # Check messages if present
        if "messages" in request_data and isinstance(request_data["messages"], list):
            for message in request_data["messages"]:
                if isinstance(message, dict) and "content" in message:
                    if isinstance(message["content"], str):
                        message_results = await self.detect(message["content"])
                        results.extend(message_results)
        
        return results
    
    def get_highest_severity(self, detections: List[InjectionDetection]) -> Optional[InjectionSeverity]:
        """
        Get the highest severity from a list of detections.
        
        Args:
            detections: List of injection detection results
            
        Returns:
            Highest severity or None if no detections
        """
        if not detections:
            return None
        
        # Map severities to numeric values
        severity_values = {
            InjectionSeverity.LOW: 1,
            InjectionSeverity.MEDIUM: 2,
            InjectionSeverity.HIGH: 3,
            InjectionSeverity.CRITICAL: 4
        }
        
        # Get highest severity
        highest = max(detections, key=lambda x: severity_values.get(x.severity, 0))
        return highest.severity
