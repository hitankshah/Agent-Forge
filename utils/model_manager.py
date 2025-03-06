"""
Utilities for managing local open source models.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import torch

logger = logging.getLogger("model_manager")

class LocalModelManager:
    """
    Manages local open source models, handling downloading, loading and inference.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = models_dir or os.environ.get("LOCAL_MODELS_DIR", "models")
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Define available local models
        self.available_models = {
            "gemma-2b": {
                "path": os.path.join(self.models_dir, "gemma-2b"),
                "hf_repo": "google/gemma-2b",
                "type": "text-generation"
            },
            "phi-2": {
                "path": os.path.join(self.models_dir, "phi-2"),
                "hf_repo": "microsoft/phi-2", 
                "type": "text-generation"
            },
            "mistral-7b": {
                "path": os.path.join(self.models_dir, "mistral-7b"),
                "hf_repo": "mistralai/Mistral-7B-v0.1",
                "type": "text-generation"
            }
        }
        
        logger.info(f"Initialized LocalModelManager with models directory: {self.models_dir}")
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check which models are available locally"""
        for name, info in self.available_models.items():
            path = info["path"]
            if os.path.exists(path) and any(os.listdir(path)):
                logger.info(f"Found local model: {name} at {path}")
                info["available"] = True
            else:
                logger.info(f"Model {name} not found locally at {path}")
                info["available"] = False
    
    def list_available_models(self) -> List[str]:
        """
        List available local models.
        
        Returns:
            List of model names
        """
        return [name for name, info in self.available_models.items() 
                if info.get("available", False)]
    
    def download_model(self, model_name: str) -> bool:
        """
        Download a model if not already available.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if successful or already downloaded
        """
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        model_info = self.available_models[model_name]
        
        # Skip if already downloaded
        if model_info.get("available", False):
            logger.info(f"Model {model_name} is already available")
            return True
            
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model {model_name} from {model_info['hf_repo']}")
            snapshot_download(
                repo_id=model_info["hf_repo"],
                local_dir=model_info["path"]
            )
            
            model_info["available"] = True
            logger.info(f"Successfully downloaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False
    
    def load_model(self, model_name: str, quantization: str = "int8") -> bool:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model to load
            quantization: Quantization format ('none', 'int8', 'int4')
            
        Returns:
            True if successful
        """
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False
            
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} is already loaded")
            return True
            
        model_info = self.available_models[model_name]
        
        # Ensure model is downloaded
        if not model_info.get("available", False):
            success = self.download_model(model_name)
            if not success:
                return False
                
        try:
            logger.info(f"Loading model {model_name}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
            
            # Load model with appropriate quantization
            load_args = {
                "local_files_only": True,
                "trust_remote_code": True
            }
            
            if self.device == "cuda":
                load_args["device_map"] = "auto"
            
            if quantization == "int8":
                load_args["load_in_8bit"] = True
            elif quantization == "int4":
                load_args["load_in_4bit"] = True
            
            model = AutoModelForCausalLM.from_pretrained(
                model_info["path"],
                **load_args
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
    
    def generate_text(self, model_name: str, prompt: str, 
                     max_length: int = 500, temperature: float = 0.7) -> Optional[str]:
        """
        Generate text using a local model.
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt
            max_length: Maximum output length
            temperature: Temperature for sampling
            
        Returns:
            Generated text or None if failed
        """
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        if model_name not in self.loaded_models:
            success = self.load_model(model_name)
            if not success:
                return None
                
        model_data = self.loaded_models[model_name]
        
        try:
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
                
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            if text.startswith(prompt):
                text = text[len(prompt):]
                
            return text.strip()
        except Exception as e:
            logger.error(f"Error generating text with model {model_name}: {str(e)}")
            return None


# Initialize once during module import
local_model_manager = LocalModelManager()
