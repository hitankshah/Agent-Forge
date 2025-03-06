"""
Agent Builder module responsible for creating agents based on descriptions
using LLM-powered analysis and generation.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
import time

from utils.llm_utils import extract_json_from_llm_response, sanitize_llm_response, PromptTemplate
from utils.error_handling import log_error, ErrorCodes, handle_exceptions

logger = logging.getLogger("AgentBuilder")

class AgentBuilder:
    """
    Builder class for creating and configuring agents using LLM-based analysis.
    """
    
    def __init__(self, model_registry=None, default_model="gpt-3.5-turbo"):
        """
        Initialize the agent builder.
        
        Args:
            model_registry: Optional model registry for accessing LLM models
            default_model: Default model to use for agent building
        """
        self.model_registry = model_registry
        self.default_model = default_model
        
    def get_model_for_task(self, task_type: str) -> str:
        """
        Determine the appropriate model for a specific task type.
        
        Args:
            task_type: Type of task (e.g., "requirement_analysis", "capability_design")
            
        Returns:
            Model name to use
        """
        # Task to model mapping
        task_models = {
            "requirement_analysis": "gpt-4" if "gpt-4" in self._get_available_models() else "gpt-3.5-turbo",
            "capability_design": "gpt-3.5-turbo",
            "evaluation": "gpt-3.5-turbo"
        }
        
        model = task_models.get(task_type, self.default_model)
        logger.info(f"Selected model {model} for task type {task_type}")
        return model
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        if self.model_registry and hasattr(self.model_registry, "list_models"):
            return self.model_registry.list_models()
        return [self.default_model]

    @handle_exceptions(error_code=ErrorCodes.JSON_PARSING_ERROR)
    def analyze_requirements(self, description: str) -> Dict[str, Any]:
        """
        Analyze user requirements and generate agent specification.
        
        Args:
            description: User description of the agent requirements
            
        Returns:
            Dict containing analyzed requirements
        """
        logger.info(f"Analyzing requirements: {description[:50]}...")
        
        # Choose appropriate model
        model = self.get_model_for_task("requirement_analysis")
        
        # Create prompt with clear instructions for JSON output
        prompt = PromptTemplate.get_prompt("REQUIREMENT_ANALYSIS", requirement=description)
        
        # Get response from LLM - this is a mock implementation
        # In a real implementation, this would call the LLM API
        response = self._mock_llm_call(prompt, model)
        
        # Clean the response
        clean_response = sanitize_llm_response(response)
        
        # Extract the JSON
        specification = extract_json_from_llm_response(clean_response)
        
        if not specification:
            error_msg = "Could not extract JSON from LLM response"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Successfully extracted agent specification")
        return specification

    @handle_exceptions(error_code=ErrorCodes.JSON_PARSING_ERROR)
    def design_capability(self, description: str) -> Dict[str, Any]:
        """
        Design a capability based on description.
        
        Args:
            description: Description of the capability
            
        Returns:
            Dict containing capability design
        """
        logger.info(f"Designing capability: {description[:50]}...")
        
        # Choose appropriate model
        model = self.get_model_for_task("capability_design")
        
        # Create prompt
        prompt = PromptTemplate.get_prompt("CAPABILITY_DESIGN", description=description)
        
        # Get response from LLM - mock implementation
        response = self._mock_llm_call(prompt, model)
        
        # Clean and extract JSON
        clean_response = sanitize_llm_response(response)
        capability = extract_json_from_llm_response(clean_response)
        
        if not capability:
            error_msg = "Could not extract capability JSON from LLM response"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info("Successfully designed capability")
        return capability

    def _mock_llm_call(self, prompt: str, model: str) -> str:
        """
        Mock LLM API call for testing purposes.
        
        Args:
            prompt: The prompt to send
            model: Model to use
            
        Returns:
            Simulated LLM response
        """
        # In a real implementation, this would call the actual LLM API
        time.sleep(0.5)  # Simulate API latency
        
        if "requirement_analysis" in prompt.lower():
            return """
```json
{
  "purpose": "Assist with research and information analysis",
  "capabilities": [
    {
      "name": "Document Analysis",
      "description": "Extract key information from research papers and articles",
      "dependencies": ["text_processing", "semantic_understanding"]
    },
    {
      "name": "Literature Search",
      "description": "Find relevant articles and papers based on keywords and topics",
      "dependencies": ["web_search", "query_formulation"]
    }
  ],
  "knowledge_domains": ["Academia", "Research Methodology", "Information Analysis"]
}
```
"""
        elif "capability_design" in prompt.lower():
            return """
```json
{
  "name": "Text Summarization",
  "description": "Create concise summaries of longer documents while preserving key information",
  "parameters": {
    "length": "adjustable",
    "focus": "key points"
  },
  "implementation_notes": "Uses extractive and abstractive summarization techniques",
  "dependencies": ["text_processing", "natural_language_understanding"]
}
```
"""
        else:
            return "{}"
