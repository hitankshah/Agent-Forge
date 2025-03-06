import logging
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from openai import OpenAI

logger = logging.getLogger('ModelProviders')

class ModelProvider(ABC):
    """Abstract base class for different model providers."""
    
    @abstractmethod
    def generate_completion(self, prompt: str, system_message: str = "", 
                           max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
        """Generate completion from the model.
        
        Returns:
            Tuple containing (response_text, metadata)
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of available models from this provider."""
        pass
    
    @abstractmethod
    def get_model_specs(self, model_name: str) -> Dict[str, Any]:
        """Return specifications for a particular model."""
        pass


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Functionality will be limited.")
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        self.default_model = default_model
        self._model_specs_cache = {}
    
    def generate_completion(self, prompt: str, system_message: str = "",
                           max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
        """Generate completion using OpenAI API."""
        if not self.api_key:
            return "API key required for completion", {"error": "No API key provided"}
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            
            metadata = {
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return f"Error: {str(e)}", {"error": str(e)}
    
    def get_available_models(self) -> List[str]:
        """Return list of available OpenAI models."""
        if not self.api_key:
            return []
            
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    def get_model_specs(self, model_name: str) -> Dict[str, Any]:
        """Return specifications for a particular OpenAI model."""
        if model_name in self._model_specs_cache:
            return self._model_specs_cache[model_name]
            
        # This is a simplified mapping as OpenAI doesn't provide detailed specs via API
        model_specs = {
            "gpt-4": {
                "context_length": 8192,
                "training_cutoff": "2023-04-01",
                "capabilities": ["text_generation", "reasoning", "code_generation"],
                "performance": {"reasoning": 0.9, "factual": 0.8, "creative": 0.9}
            },
            "gpt-3.5-turbo": {
                "context_length": 4096,
                "training_cutoff": "2022-09-01",
                "capabilities": ["text_generation", "reasoning", "code_generation"],
                "performance": {"reasoning": 0.7, "factual": 0.7, "creative": 0.8}
            }
        }
        
        # Default for unknown models
        if model_name not in model_specs:
            model_specs[model_name] = {
                "context_length": 2048,
                "capabilities": ["text_generation"],
                "performance": {"reasoning": 0.5, "factual": 0.5, "creative": 0.5}
            }
            
        self._model_specs_cache[model_name] = model_specs[model_name]
        return model_specs[model_name]


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models (Claude)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-2"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Functionality will be limited.")
            
        self.default_model = default_model
        self._model_specs_cache = {
            "claude-2": {
                "context_length": 100000,
                "training_cutoff": "2023-01-01",
                "capabilities": ["text_generation", "reasoning", "code_generation"],
                "performance": {"reasoning": 0.85, "factual": 0.8, "creative": 0.8}
            },
            "claude-instant": {
                "context_length": 100000,
                "training_cutoff": "2022-12-01",
                "capabilities": ["text_generation", "reasoning"],
                "performance": {"reasoning": 0.7, "factual": 0.7, "creative": 0.7}
            }
        }
    
    def generate_completion(self, prompt: str, system_message: str = "",
                           max_tokens: int = 1000, temperature: float = 0.7) -> Tuple[str, Dict[str, Any]]:
        """Generate completion using Anthropic API."""
        if not self.api_key:
            return "API key required for completion", {"error": "No API key provided"}
            
        # Note: This is a placeholder. Actual implementation would use the Anthropic API
        logger.warning("Anthropic API not fully implemented")
        return f"Sample response from {self.default_model}", {"model": self.default_model}
    
    def get_available_models(self) -> List[str]:
        """Return list of available Anthropic models."""
        if not self.api_key:
            return []
        
        # This would normally query the Anthropic API
        return ["claude-2", "claude-instant"]
    
    def get_model_specs(self, model_name: str) -> Dict[str, Any]:
        """Return specifications for a particular Anthropic model."""
        if model_name in self._model_specs_cache:
            return self._model_specs_cache[model_name]
            
        # Default for unknown models
        return {
            "context_length": 2048,
            "capabilities": ["text_generation"],
            "performance": {"reasoning": 0.5, "factual": 0.5, "creative": 0.5}
        }


class ModelRegistry:
    """Registry for managing and selecting models."""
    
    def __init__(self):
        self.providers = {}
        self.performance_data = {}
        
    def register_provider(self, name: str, provider: ModelProvider) -> None:
        """Register a model provider."""
        self.providers[name] = provider
        
    def select_best_model(self, task_type: str, min_performance: float = 0.7) -> Tuple[str, ModelProvider]:
        """Select the best model for a given task type."""
        best_score = 0.0
        best_model = None
        best_provider = None
        
        for provider_name, provider in self.providers.items():
            models = provider.get_available_models()
            for model_name in models:
                specs = provider.get_model_specs(model_name)
                
                # Check if model has performance data for this task
                performance = specs.get("performance", {}).get(task_type, 0.0)
                
                # Check if model performance exceeds minimum and is better than current best
                if performance >= min_performance and performance > best_score:
                    best_score = performance
                    best_model = model_name
                    best_provider = provider
        
        if best_model:
            return best_model, best_provider
        else:
            # Fall back to default if no suitable model found
            fallback_provider = next(iter(self.providers.values()))
            return fallback_provider.default_model, fallback_provider
    
    def update_model_performance(self, provider_name: str, model_name: str, 
                               task_type: str, score: float) -> None:
        """Update performance data for a model."""
        if provider_name not in self.performance_data:
            self.performance_data[provider_name] = {}
        
        if model_name not in self.performance_data[provider_name]:
            self.performance_data[provider_name][model_name] = {}
        
        # Update using exponential moving average
        current = self.performance_data[provider_name][model_name].get(task_type, score)
        self.performance_data[provider_name][model_name][task_type] = 0.8 * current + 0.2 * score
        
    def save_performance_data(self, filepath: str) -> bool:
        """Save performance data to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
            return False
    
    def load_performance_data(self, filepath: str) -> bool:
        """Load performance data from file."""
        try:
            with open(filepath, 'r') as f:
                self.performance_data = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return False
