import os
import yaml
from typing import Dict, Any, Optional
from loguru import logger
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Default configuration
    default_config = {
        "llm": {
            "provider": "openai",
            "openai": {
                "model": "gpt-3.5-turbo"
            }
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002"
        },
        "agent_defaults": {
            "memory_enabled": True,
            "tools_enabled": True,
            "max_history_tokens": 2000
        }
    }
    
    # Try to load configuration file
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config
        else:
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            
            # Write default config to file
            try:
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default configuration file at {config_file}")
            except Exception as e:
                logger.error(f"Failed to create default configuration file: {str(e)}")
            
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return default_config

def save_config(config: Dict[str, Any], config_file: str = "config.yaml") -> bool:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

def update_config(updates: Dict[str, Any], config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Update configuration with new values and save to file.
    
    Args:
        updates: Dictionary of configuration updates
        config_file: Path to the configuration file
        
    Returns:
        Updated configuration dictionary
    """
    # Load current configuration
    config = load_config(config_file)
    
    # Update configuration recursively
    def update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict_recursive(d[k], v)
            else:
                d[k] = v
        return d
    
    # Apply updates
    config = update_dict_recursive(config, updates)
    
    # Save updated configuration
    save_config(config, config_file)
    
    return config

def get_env_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    
    Returns:
        Dictionary with configuration from environment variables
    """
    env_config = {}
    
    # LLM configuration
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider:
        env_config["llm"] = {"provider": llm_provider}
        
        # Provider-specific model
        model_var = f"{llm_provider.upper()}_MODEL"
        model = os.getenv(model_var)
        if model:
            if llm_provider not in env_config["llm"]:
                env_config["llm"][llm_provider] = {}
            env_config["llm"][llm_provider]["model"] = model
    
    # Embedding configuration
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if embedding_model:
        if "embedding" not in env_config:
            env_config["embedding"] = {}
        env_config["embedding"]["model"] = embedding_model
        
        # Try to determine provider from model name
        if "openai" in embedding_model or embedding_model.startswith("text-embedding"):
            env_config["embedding"]["provider"] = "openai"
        elif "cohere" in embedding_model:
            env_config["embedding"]["provider"] = "cohere"
        elif "huggingface" in embedding_model or "/" in embedding_model:
            env_config["embedding"]["provider"] = "huggingface"
    
    return env_config

def merge_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration with environment variables.
    Environment variables take precedence.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    env_config = get_env_config()
    
    # Update configuration recursively
    def update_dict_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict_recursive(d[k], v)
            else:
                d[k] = v
        return d
    
    # Apply updates from environment
    return update_dict_recursive(config, env_config)

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "agent_defaults": {
            "memory_enabled": True,
            "tools_enabled": True,
            "max_history_tokens": 2000
        },
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            }
        },
        "embedding": {
            "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
            "model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        }
    }

def get_provider_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """
    Get provider-specific configuration.
    
    Args:
        config: Main configuration dictionary
        provider: Provider name
        
    Returns:
        Provider configuration
    """
    # Get LLM configuration
    llm_config = config.get("llm", {})
    
    # Get provider configuration
    provider_config = llm_config.get(provider, {})
    
    # Add any environment variables
    if provider == "openai":
        provider_config["api_key"] = os.getenv("OPENAI_API_KEY", "")
    elif provider == "anthropic":
        provider_config["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    elif provider == "cohere":
        provider_config["api_key"] = os.getenv("COHERE_API_KEY", "")
    
    return provider_config

def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Deep merge two dictionaries. The source dictionary values
    override the target dictionary values where keys overlap.
    
    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value