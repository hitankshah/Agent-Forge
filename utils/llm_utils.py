"""
Utilities for working with LLM responses, particularly for robust JSON extraction.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger("llm_utils")

def extract_json_from_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract JSON from an LLM response.
    
    Args:
        response: The full text response from an LLM
        
    Returns:
        Extracted JSON as a Python dict, or None if extraction failed
    """
    if not response:
        logger.error("Empty response received")
        return None
    
    # First sanitize the response
    response = sanitize_llm_response(response)
        
    # Try methods in order of likelihood
    extraction_methods = [
        _extract_json_from_code_blocks,
        _extract_json_with_brackets,
        _extract_json_with_repair,
        _extract_json_from_raw_text
    ]
    
    for method in extraction_methods:
        try:
            result = method(response)
            if result:
                return result
        except Exception as e:
            logger.debug(f"JSON extraction method {method.__name__} failed: {str(e)}")
            continue
    
    logger.error("All JSON extraction methods failed")
    logger.debug(f"Failed response preview: {response[:300]}...")
    return None

def _extract_json_from_code_blocks(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from code blocks in text."""
    # Look for content in triple backticks
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)
    return None

def _extract_json_with_brackets(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text with curly brackets."""
    pattern = r"\{[\s\S]*?\}"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    return None

def _extract_json_with_repair(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to repair and extract JSON from text."""
    try:
        # Attempt to repair common JSON issues
        text = re.sub(r"(\w+):", r'"\1":', text)  # Add quotes around keys
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def _extract_json_from_raw_text(text: str) -> Optional[Dict[str, Any]]: 
    """Extract JSON from raw text.""" 
    try: 
        return json.loads(text) 
    except json.JSONDecodeError: 
        return None

def sanitize_llm_response(response: str) -> str:
    """
    Sanitize an LLM response to remove problematic elements.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Sanitized response string
    """
    # Remove any HTML-like content that might cause issues
    response = re.sub(r'<div.*?>|</div>|<p.*?>|</p>|<h\d.*?>|</h\d>', '', response)
    
    # Clean up any markdown artifacts that might interfere with JSON parsing
    response = re.sub(r'^\s*#.*$', '', response, flags=re.MULTILINE)  # Remove markdown headers
    
    return response.strip()

def create_structured_prompt(task: str, content: str) -> str:
    """
    Create a prompt that encourages structured JSON responses.
    
    Args:
        task: The task description
        content: The main content for the prompt
        
    Returns:
        A formatted prompt string
    """
    return f"""
{task}

{content}

Please provide your response in valid JSON format.
Format your response as follows:

```json
{{
    "key1": "value1",
    "key2": "value2"
}}
```

Ensure the JSON is complete and properly formatted.
"""

class PromptTemplate:
    """Class for managing prompt templates with JSON output enforcement"""
    
    REQUIREMENT_ANALYSIS = """
You are analyzing user requirements for an AI agent.
Provide the following information in your JSON response:

```json
{
  "purpose": "Brief statement of the agent's main purpose",
  "capabilities": [
    {
      "name": "capability_name",
      "description": "detailed description",
      "dependencies": ["dep1", "dep2"]
    }
  ],
  "knowledge_domains": ["domain1", "domain2"]
}
```

User requirement: {requirement}
"""

    CAPABILITY_DESIGN = """
Design a capability based on the following description.
Provide the following information in your JSON response:

```json
{
  "name": "capability_name",
  "description": "detailed description",
  "parameters": {
    "param1": "value1"
  },
  "implementation_notes": "notes on implementation",
  "dependencies": ["dep1", "dep2"]
}
```

Capability description: {description}
"""

    @staticmethod
    def get_prompt(template_name: str, **kwargs) -> str:
        """Get a formatted prompt template"""
        template = getattr(PromptTemplate, template_name, None)
        if not template:
            raise ValueError(f"Unknown template name: {template_name}")
        return template.format(**kwargs)
