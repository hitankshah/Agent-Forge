"""
Helper functions for loading and using open source models.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import torch

# Configure logging
logger = logging.getLogger("open_source_models")

class OpenSourceModelManager:
    """
    Class for managing open source LLM models.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the open source model manager.
        
        Args:
            model_config: Configuration for available models
        """
        self.model_config = model_config or {}
        self.loaded_models = {}
        self.device = "cpu"
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA is available, models will use GPU by default")
        else:
            logger.info("CUDA is not available, models will use CPU")
    
    def list_available_models(self) -> List[str]:
        """
        List available open source models.
        
        Returns:
            List of model names
        """
        return list(self.model_config.keys())
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            bool: True if successful
        """
        if model_name not in self.model_config:
            logger.error(f"Model {model_name} not found in configuration")
            return False
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} is already loaded")
            return True
        
        try:
            model_info = self.model_config[model_name]
            model_path = model_info["path"]
            device = model_info.get("device", self.device)
            quantization = model_info.get("quantization", None)
            
            logger.info(f"Loading model {model_name} from {model_path}")
            
            # Import transformers only when needed
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                logger.error("transformers package not installed. Please install with: pip install transformers")
                return False
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model with quantization if specified
            if quantization == "int8":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            elif quantization == "int4":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    load_in_4bit=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True
                )
            
            # Store model and tokenizer
            self.loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            logger.info(f"Successfully loaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False

    def generate_text(self, model_name: str, prompt: str, max_length: int = 100, 
                      temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using a loaded model.
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text as string
        """
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return f"Error: Could not load model {model_name}"
        
        try:
            model_data = self.loaded_models[model_name]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
                
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text with model {model_name}: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            bool: True if successful
        """
        if model_name not in self.loaded_models:
            logger.info(f"Model {model_name} is not loaded")
            return True
            
        try:
            # Remove references to model and tokenizer
            del self.loaded_models[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"Unloaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
            
def get_open_source_model_manager(config: Optional[Dict[str, Any]] = None) -> OpenSourceModelManager:
    """
    Create and configure an OpenSourceModelManager from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OpenSourceModelManager instance
    """
    if not config:
        return OpenSourceModelManager({})
        
    model_config = config.get("open_source_models", {})
    return OpenSourceModelManager(model_config)