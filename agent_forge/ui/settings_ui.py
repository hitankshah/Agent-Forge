import streamlit as st
import os
import json
import yaml
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv, set_key
import re
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_forge.integrations.embeddings import EmbeddingConfig

class SettingsUI:
    """Professional UI for managing settings, environment variables and configurations."""
    
    def __init__(self):
        """Initialize the settings UI manager."""
        self.env_path = find_dotenv(usecwd=True)
        if not self.env_path:
            # Create a default .env file in the project root if none exists
            self.env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), ".env")
            
        # Load environment variables
        load_dotenv(self.env_path)
        
        # Define common model providers and their configurations
        self.providers = {
            "openai": {
                "name": "OpenAI",
                "env_vars": ["OPENAI_API_KEY"],
                "models": {
                    "embedding": ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                    "completion": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
                }
            },
            "cohere": {
                "name": "Cohere",
                "env_vars": ["COHERE_API_KEY"],
                "models": {
                    "embedding": ["embed-english-v3.0", "embed-multilingual-v3.0"],
                    "completion": ["command", "command-light", "command-r", "command-r-plus"]
                }
            },
            "huggingface": {
                "name": "Hugging Face",
                "env_vars": ["HUGGINGFACE_API_KEY"],
                "models": {
                    "embedding": [
                        "sentence-transformers/all-mpnet-base-v2",
                        "sentence-transformers/all-MiniLM-L6-v2", 
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    ],
                    "completion": []  # Will be populated later for Transformers
                }
            },
            "anthropic": {
                "name": "Anthropic",
                "env_vars": ["ANTHROPIC_API_KEY"],
                "models": {
                    "embedding": [],
                    "completion": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
                }
            }
        }

    def render_env_manager(self):
        """Render the environment variables manager UI."""
        st.subheader("Environment Variables Manager")
        
        # Display current environment variables
        current_env = self._load_env_vars()
        sensitive_keys = ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Current Environment Variables")
            
            # Group by provider
            provider_vars = {}
            for key, value in current_env.items():
                provider = "Other"
                for p_key, p_data in self.providers.items():
                    if any(key.startswith(env_prefix) for env_prefix in [p_key.upper()] + [var.split("_")[0] for var in p_data["env_vars"]]):
                        provider = p_data["name"]
                        break
                
                if provider not in provider_vars:
                    provider_vars[provider] = {}
                    
                # Mask sensitive values
                masked_value = value
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    if value and len(value) > 4:
                        masked_value = value[:4] + "•" * (len(value) - 4)
                    else:
                        masked_value = "•" * len(value) if value else ""
                
                provider_vars[provider][key] = masked_value
            
            # Display grouped variables
            for provider, variables in provider_vars.items():
                with st.expander(f"{provider} ({len(variables)} variables)", expanded=provider != "Other"):
                    for key, masked_value in variables.items():
                        st.text_input(key, value=masked_value, disabled=True, key=f"env_display_{key}")
        
        with col2:
            st.markdown("### Actions")
            if st.button("Refresh Variables", use_container_width=True):
                st.experimental_rerun()
                
            if st.button("Download .env File", use_container_width=True):
                with open(self.env_path, 'r') as f:
                    env_content = f.read()
                    
                st.download_button(
                    label="Download",
                    data=env_content,
                    file_name=".env",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Add or update environment variables
        st.markdown("### Add or Update Environment Variable")
        col1, col2 = st.columns([2, 1])
        with col1:
            new_key = st.text_input("Variable Name", key="new_env_key")
        with col2:
            is_secret = st.checkbox("Is Secret/API Key", value=True, key="is_secret_env")
        
        new_value = st.text_input("Variable Value", type="password" if is_secret else "text", key="new_env_value")
        
        if st.button("Save Variable", use_container_width=True):
            if new_key and new_value:
                self._update_env_var(new_key, new_value)
                st.success(f"Updated environment variable: {new_key}")
                # Clear the input fields
                st.session_state.new_env_key = ""
                st.session_state.new_env_value = ""
                # Force refresh
                st.experimental_rerun()
            else:
                st.error("Both variable name and value are required")
        
        # Bulk edit environment variables
        with st.expander("Bulk Edit Environment Variables"):
            st.markdown("""
            Edit multiple environment variables at once. Format:
            ```
            VARIABLE_NAME=value
            ANOTHER_VARIABLE=another_value
            ```
            **Note:** This will add or update the specified variables. Existing variables not included here will remain unchanged.
            """)
            
            bulk_env_content = st.text_area(
                "Environment Variables (one per line)",
                height=200,
                key="bulk_env_content"
            )
            
            if st.button("Save All Variables", use_container_width=True):
                if bulk_env_content:
                    # Parse variables
                    var_count = 0
                    lines = bulk_env_content.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and "=" in line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                                
                            if key:
                                self._update_env_var(key, value)
                                var_count += 1
                    
                    if var_count > 0:
                        st.success(f"Updated {var_count} environment variables")
                        # Force refresh
                        st.experimental_rerun()
                    else:
                        st.warning("No valid environment variables found")
                else:
                    st.error("No content provided")

    def render_model_configuration(self):
        """Render the model configuration UI for embedding and LLM models."""
        st.subheader("AI Model Configuration")
        
        tab1, tab2 = st.tabs(["Embedding Models", "LLM Models"])
        
        with tab1:
            self._render_embedding_model_config()
            
        with tab2:
            self._render_llm_model_config()

    def _render_embedding_model_config(self):
        """Render the embedding model configuration UI."""
        st.markdown("### Embedding Model Settings")
        
        # Current default model
        current_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Model provider selection
        provider_options = ["openai", "cohere", "huggingface", "custom"]
        
        # Determine current provider
        current_provider = "custom"
        for provider in provider_options[:3]:  # Exclude custom
            provider_models = self.providers[provider]["models"]["embedding"]
            if any(model == current_model or current_model.endswith(f"/{model}") for model in provider_models):
                current_provider = provider
                break
        
        # Provider selection
        selected_provider = st.selectbox(
            "Provider", 
            options=provider_options,
            format_func=lambda x: self.providers.get(x, {}).get("name", x.capitalize()),
            index=provider_options.index(current_provider)
        )
        
        # Model selection based on provider
        if selected_provider == "custom":
            custom_model = st.text_input(
                "Custom Model Name", 
                value=current_model if current_provider == "custom" else ""
            )
            selected_model = custom_model
        else:
            provider_models = self.providers[selected_provider]["models"]["embedding"]
            
            # Check if required API key is set
            required_vars = self.providers[selected_provider]["env_vars"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                st.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
                st.markdown(f"Set these variables in the **Environment Variables Manager** section.")
            
            # Model selection
            selected_model = st.selectbox(
                "Model",
                options=provider_models,
                index=0
            )
            
            # If HuggingFace, prefix with provider namespace
            if selected_provider == "huggingface" and "/" not in selected_model:
                selected_model = f"sentence-transformers/{selected_model}"
        
        # Advanced configuration
        with st.expander("Advanced Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.slider(
                    "Batch Size", 
                    min_value=1, 
                    max_value=64, 
                    value=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
                    help="Number of texts to process at once"
                )
                
            with col2:
                normalize = st.checkbox(
                    "Normalize Embeddings", 
                    value=os.getenv("EMBEDDING_NORMALIZE", "").lower() == "true",
                    help="Normalize embedding vectors (L2 norm = 1)"
                )
            
            # Device selection for HuggingFace models
            if selected_provider == "huggingface":
                try:
                    import torch
                    device_options = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                    
                    current_device = os.getenv("EMBEDDING_DEVICE", "")
                    if not current_device:
                        current_device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                    device = st.selectbox(
                        "Device",
                        options=device_options,
                        index=0 if current_device == "cpu" else 1,
                        help="Processing device for embedding model"
                    )
                except ImportError:
                    st.warning("PyTorch not installed. Using CPU for embeddings.")
                    device = "cpu"
            else:
                device = "cpu"  # Not used for API-based embeddings
        
        # Save configuration
        if st.button("Save Embedding Configuration", use_container_width=True):
            # Update environment variables
            self._update_env_var("EMBEDDING_MODEL", selected_model)
            self._update_env_var("EMBEDDING_BATCH_SIZE", str(batch_size))
            self._update_env_var("EMBEDDING_NORMALIZE", str(normalize).lower())
            
            if selected_provider == "huggingface":
                self._update_env_var("EMBEDDING_DEVICE", device)
            
            st.success("Embedding configuration saved")
            st.experimental_rerun()

    def _render_llm_model_config(self):
        """Render the LLM model configuration UI."""
        st.markdown("### LLM Model Settings")
        
        # Current default model and provider
        current_provider = os.getenv("LLM_PROVIDER", "openai")
        current_model = os.getenv(f"{current_provider.upper()}_MODEL", "")
        
        if not current_model and current_provider in self.providers:
            # Use first completion model as default if available
            completion_models = self.providers[current_provider]["models"]["completion"]
            if completion_models:
                current_model = completion_models[0]
        
        # Provider selection
        provider_options = list(self.providers.keys()) + ["custom"]
        
        selected_provider = st.selectbox(
            "Provider", 
            options=provider_options,
            format_func=lambda x: self.providers.get(x, {}).get("name", x.capitalize()),
            index=provider_options.index(current_provider) if current_provider in provider_options else 0
        )
        
        # Model selection based on provider
        if selected_provider == "custom":
            custom_model = st.text_input(
                "Custom Model Name", 
                value=current_model if current_provider == "custom" else ""
            )
            selected_model = custom_model
            
            # Custom provider settings
            custom_base_url = st.text_input(
                "API Base URL",
                value=os.getenv("CUSTOM_BASE_URL", ""),
                help="The base URL for API requests (e.g., http://localhost:8000/v1)"
            )
            
            api_type_options = ["openai_compatible", "anthropic_compatible", "raw_completion"]
            custom_api_type = st.selectbox(
                "API Type",
                options=api_type_options,
                index=api_type_options.index(os.getenv("CUSTOM_API_TYPE", "openai_compatible")) 
                      if os.getenv("CUSTOM_API_TYPE", "") in api_type_options else 0,
                help="The format of the API requests/responses"
            )
        else:
            # Check if required API key is set
            required_vars = self.providers[selected_provider]["env_vars"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                st.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
                st.markdown(f"Set these variables in the **Environment Variables Manager** section.")
            
            # Model selection
            provider_models = self.providers[selected_provider]["models"]["completion"]
            
            if provider_models:
                model_idx = 0
                if current_model in provider_models:
                    model_idx = provider_models.index(current_model)
                    
                selected_model = st.selectbox(
                    "Model",
                    options=provider_models,
                    index=model_idx
                )
            else:
                selected_model = st.text_input(
                    "Model Name", 
                    value=current_model
                )
        
        # Advanced configuration
        with st.expander("Advanced Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                    step=0.1,
                    help="Controls randomness: lower values are more deterministic"
                )
                
                max_tokens = st.number_input(
                    "Max Output Tokens", 
                    min_value=1, 
                    max_value=32000, 
                    value=int(os.getenv("LLM_MAX_TOKENS", "1000")),
                    help="Maximum number of tokens to generate"
                )
            
            with col2:
                top_p = st.slider(
                    "Top P", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=float(os.getenv("LLM_TOP_P", "0.95")),
                    help="Controls diversity via nucleus sampling"
                )
                
                frequency_penalty = st.slider(
                    "Frequency Penalty", 
                    min_value=0.0, 
                    max_value=2.0, 
                    value=float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")),
                    help="Reduces repetition of token sequences"
                )
        
        # System prompt
        system_prompt = st.text_area(
            "Default System Prompt",
            value=os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful AI assistant."),
            height=100,
            help="The system prompt that sets the behavior of the model"
        )
        
        # Save configuration
        if st.button("Save LLM Configuration", use_container_width=True):
            # Update environment variables
            self._update_env_var("LLM_PROVIDER", selected_provider)
            self._update_env_var(f"{selected_provider.upper()}_MODEL", selected_model)
            self._update_env_var("LLM_TEMPERATURE", str(temperature))
            self._update_env_var("LLM_MAX_TOKENS", str(int(max_tokens)))
            self._update_env_var("LLM_TOP_P", str(top_p))
            self._update_env_var("LLM_FREQUENCY_PENALTY", str(frequency_penalty))
            self._update_env_var("LLM_SYSTEM_PROMPT", system_prompt)
            
            # Save custom provider settings if applicable
            if selected_provider == "custom":
                self._update_env_var("CUSTOM_BASE_URL", custom_base_url)
                self._update_env_var("CUSTOM_API_TYPE", custom_api_type)
            
            st.success("LLM configuration saved")
            st.experimental_rerun()

    def render_advanced_settings(self):
        """Render the advanced settings UI."""
        st.subheader("Advanced Settings")
        
        tab1, tab2 = st.tabs(["Agent Configuration", "System Settings"])
        
        with tab1:
            self._render_agent_settings()
        
        with tab2:
            self._render_system_settings()

    def _render_agent_settings(self):
        """Render the agent configuration settings."""
        st.markdown("### Agent Configuration")
        
        # Try to load agent config
        config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Allow editing the YAML directly
                with st.expander("Edit Configuration YAML"):
                    config_yaml = st.text_area(
                        "Configuration YAML", 
                        value=yaml.dump(config, default_flow_style=False),
                        height=400
                    )
                    
                    if st.button("Save YAML Configuration", use_container_width=True):
                        try:
                            # Parse to validate
                            new_config = yaml.safe_load(config_yaml)
                            
                            # Write to file
                            with open(config_path, 'w') as f:
                                f.write(config_yaml)
                                
                            st.success("Configuration saved")
                        except Exception as e:
                            st.error(f"Error saving configuration: {str(e)}")
                
                # Visual editor for key settings
                st.markdown("#### Visual Configuration Editor")
                
                # Agent defaults
                st.markdown("##### Default Agent Settings")
                
                agent_defaults = config.get("agent_defaults", {})
                col1, col2 = st.columns(2)
                
                with col1:
                    memory_enabled = st.checkbox(
                        "Enable Memory", 
                        value=agent_defaults.get("memory_enabled", True),
                        help="Agents remember conversation history"
                    )
                
                with col2:
                    tools_enabled = st.checkbox(
                        "Enable Tools", 
                        value=agent_defaults.get("tools_enabled", True),
                        help="Agents can use tools like web search"
                    )
                
                # Max history tokens
                max_history = st.number_input(
                    "Max History Tokens", 
                    value=agent_defaults.get("max_history_tokens", 2000),
                    help="Maximum tokens to use for conversation history"
                )
                
                # Save agent defaults
                if st.button("Save Agent Defaults", use_container_width=True):
                    # Update config
                    if "agent_defaults" not in config:
                        config["agent_defaults"] = {}
                        
                    config["agent_defaults"]["memory_enabled"] = memory_enabled
                    config["agent_defaults"]["tools_enabled"] = tools_enabled
                    config["agent_defaults"]["max_history_tokens"] = int(max_history)
                    
                    # Save to file
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                        
                    st.success("Agent defaults saved")
                
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
        else:
            st.warning(f"Configuration file not found: {config_path}")
            
            if st.button("Create Default Configuration"):
                # Create default config
                default_config = {
                    "agent_defaults": {
                        "memory_enabled": True,
                        "tools_enabled": True,
                        "max_history_tokens": 2000
                    },
                    "llm": {
                        "provider": "openai",
                        "openai": {
                            "model": "gpt-3.5-turbo"
                        }
                    },
                    "embedding": {
                        "provider": "openai",
                        "model": "text-embedding-ada-002"
                    }
                }
                
                # Write to file
                try:
                    with open(config_path, 'w') as f:
                        yaml.dump(default_config, f, default_flow_style=False)
                        
                    st.success(f"Default configuration created at {config_path}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating configuration: {str(e)}")

    def _render_system_settings(self):
        """Render system-wide settings."""
        st.markdown("### System Settings")
        
        # Log level
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        current_log_level = os.getenv("LOG_LEVEL", "INFO")
        
        selected_log_level = st.selectbox(
            "Log Level",
            options=log_levels,
            index=log_levels.index(current_log_level) if current_log_level in log_levels else 1,
            help="Controls verbosity of logging"
        )
        
        # Cache settings
        col1, col2 = st.columns(2)
        
        with col1:
            cache_enabled = st.checkbox(
                "Enable Caching", 
                value=os.getenv("CACHE_ENABLED", "").lower() != "false",
                help="Cache API responses to reduce costs and latency"
            )
        
        with col2:
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                value=int(os.getenv("CACHE_TTL", "3600")),
                help="Time-to-live for cached items"
            )
        
        # Save system settings
        if st.button("Save System Settings", use_container_width=True):
            self._update_env_var("LOG_LEVEL", selected_log_level)
            self._update_env_var("CACHE_ENABLED", str(cache_enabled).lower())
            self._update_env_var("CACHE_TTL", str(int(cache_ttl)))
            
            st.success("System settings saved")
            st.experimental_rerun()

    def render_embedding_demo(self):
        """Render a demonstration UI for testing embeddings."""
        st.subheader("Embedding Model Demo")
        
        # Text input
        demo_text = st.text_area(
            "Enter text to embed",
            value="This is a test sentence to demonstrate the embedding functionality.",
            height=100
        )
        
        # Model selection
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_default = st.checkbox("Use Default Model", value=True)
        
        # Allow selecting a different model
        if not use_default:
            with col2:
                custom_model = st.text_input("Custom Model", value=embedding_model)
                embedding_model = custom_model
        
        # Generate embedding button
        if st.button("Generate Embedding", use_container_width=True):
            if demo_text:
                try:
                    import sys
                    import numpy as np
                    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
                    from agent_forge.integrations.embeddings import get_embedding
                    
                    with st.spinner("Generating embedding..."):
                        # Create config from environment variables
                        config = EmbeddingConfig(
                            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
                            device=os.getenv("EMBEDDING_DEVICE", None),
                            normalize_embeddings=os.getenv("EMBEDDING_NORMALIZE", "").lower() == "true"
                        )
                        
                        # Get embedding
                        start_time = time.time()
                        embedding = get_embedding(demo_text, model=embedding_model, config=config)
                        elapsed_time = time.time() - start_time
                        
                        # Show results
                        st.success(f"Embedding generated in {elapsed_time:.2f} seconds")
                        
                        # Display embedding stats
                        embedding_array = np.array(embedding)
                        
                        st.markdown(f"**Embedding dimension:** {len(embedding)}")
                        st.markdown(f"**L2 norm:** {np.linalg.norm(embedding_array):.4f}")
                        st.markdown(f"**Min value:** {embedding_array.min():.4f}")
                        st.markdown(f"**Max value:** {embedding_array.max():.4f}")
                        st.markdown(f"**Mean value:** {embedding_array.mean():.4f}")
                        
                        # Visualize first 20 dimensions
                        st.markdown("#### First 20 dimensions")
                        st.bar_chart(embedding[:20])
                        
                        # Option to download full embedding
                        embedding_json = json.dumps(embedding)
                        st.download_button(
                            "Download Embedding (JSON)",
                            embedding_json,
                            file_name="embedding.json",
                            mime="application/json"
                        )
                        
                except Exception as e:
                    st.error(f"Error generating embedding: {str(e)}")
            else:
                st.warning("Please enter text to embed")

    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}
        
        # First load from OS environment
        for key, value in os.environ.items():
            env_vars[key] = value
        
        # Then load from .env file to override
        if os.path.exists(self.env_path):
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        env_vars[key] = value
        
        return env_vars

    def _update_env_var(self, key: str, value: str):
        """Update an environment variable in the .env file."""
        # Create .env file if it doesn't exist
        env_dir = os.path.dirname(self.env_path)
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)
            
        if not os.path.exists(self.env_path):
            with open(self.env_path, 'w') as f:
                f.write("# Environment variables for Agent Forge\n\n")
        
        # Update the environment variable
        set_key(self.env_path, key, value)
        
        # Also update in current process
        os.environ[key] = value

def render_settings_page():
    """Render the complete settings page."""
    st.set_page_config(
        page_title="Agent Forge Settings",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    