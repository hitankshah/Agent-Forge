"""
Script to download free LLM models for the Agent Platform.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("download_models")

def download_model(model_name: str, save_path: str) -> bool:
    """
    Download a model from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier
        save_path: Path to save the model
        
    Returns:
        True if successful
    """
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading model {model_name} to {save_path}")
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"]  # Optional: Reduce size
        )
        logger.info(f"Model {model_name} downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        return False

def get_model_list_from_config() -> List[Dict[str, Any]]:
    """
    Get list of models to download from config.
    
    Returns:
        List of model configurations
    """
    try:
        # Look for config in standard locations
        config_paths = [
            os.path.join(os.getcwd(), "config", "integration_config.json"),
            os.path.join(os.getcwd(), "integration_config.json"),
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return [
                        {
                            "name": model_name,
                            "hf_repo": model_info["path"],
                            "save_path": os.path.join("models", model_name),
                            "priority": model_info.get("priority", 0)
                        }
                        for model_name, model_info in config.get("open_source_models", {}).items()
                    ]
    
        # If no config found, use default models
        logger.warning("No configuration found. Using default models.")
        return [
            {
                "name": "phi-2",
                "hf_repo": "microsoft/phi-2",
                "save_path": "models/phi-2",
                "priority": 1
            },
            {
                "name": "gemma-2b",
                "hf_repo": "google/gemma-2b",
                "save_path": "models/gemma-2b",
                "priority": 2
            }
        ]
    except Exception as e:
        logger.error(f"Error reading model list: {str(e)}")
        return []

def main():
    """Main function to download models."""
    parser = argparse.ArgumentParser(description="Download free LLM models")
    parser.add_argument("--models", nargs="+", help="List of model names to download")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")
    parser.add_argument("--from-config", action="store_true", help="Read models from config")
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    if args.from_config:
        models_to_download = get_model_list_from_config()
    elif args.models:
        models_to_download = [
            {"name": model, "hf_repo": model, "save_path": os.path.join(args.models_dir, model.split("/")[-1])}
            for model in args.models
        ]
    else:
        # Default models if no arguments provided
        models_to_download = get_model_list_from_config()
    
    logger.info(f"Will download {len(models_to_download)} models")
    
    # Download models
    for model_info in models_to_download:
        download_model(model_info["hf_repo"], model_info["save_path"])

if __name__ == "__main__":
    main()
