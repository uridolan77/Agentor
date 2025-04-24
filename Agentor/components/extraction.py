from typing import Dict, Any, TypeVar, Generic, Type, Optional
from pydantic import BaseModel, ValidationError
import json
import re
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class StructuredOutputParser(Generic[T]):
    """Extract structured data from LLM responses using Pydantic models."""
    
    def __init__(self, model_class: Type[T]):
        """Initialize the structured output parser.
        
        Args:
            model_class: The Pydantic model class to extract
        """
        self.model_class = model_class
    
    async def parse(self, text: str) -> Optional[T]:
        """Parse structured data from text.
        
        Args:
            text: The text to extract from
            
        Returns:
            The parsed data or None if parsing failed
        """
        # Try to extract JSON from the text
        json_str = self._extract_json(text)
        
        if not json_str:
            logger.warning("No JSON found in text")
            return None
        
        # Parse the JSON
        try:
            data = json.loads(json_str)
            # Validate and convert to the model
            return self.model_class(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse structured data: {str(e)}")
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text.
        
        Args:
            text: The text to extract from
            
        Returns:
            The extracted JSON string or None if not found
        """
        # Try to find JSON in code blocks first
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # JSON in code blocks with json tag
            r'```\s*([\{\[][\s\S]*?[\}\]])\s*```',  # JSON in code blocks without tag
            r'(\{[\s\S]*\})',  # JSON object
            r'(\[[\s\S]*\])'   # JSON array
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None


class OutputFormatter:
    """Format LLM prompts for structured output."""
    
    @staticmethod
    def format_prompt_for_structured_output(prompt: str, model_class: Type[BaseModel]) -> str:
        """Format a prompt for structured output.
        
        Args:
            prompt: The base prompt
            model_class: The Pydantic model class for the expected output
            
        Returns:
            A formatted prompt
        """
        schema = model_class.schema()
        
        schema_str = json.dumps(schema, indent=2)
        
        formatted_prompt = f"""
{prompt}

Return your answer as a valid JSON object that conforms to this schema:
```
{schema_str}
```

Ensure your response is a valid JSON object and nothing else.
"""
        return formatted_prompt