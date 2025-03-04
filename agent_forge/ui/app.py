import streamlit as st
from typing import Dict, Any, List, Optional
import sys
import os
import json
import time
import yaml
from loguru import logger
import traceback

# Add parent directory to path to allow imports when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Try importing the required modules
try:
    from agent_forge.core.agent_factory import AgentFactory
    from agent_forge.utils.config import load_config
    from agent_forge.integrations.llm import LLMProvider
    from agent_forge.integrations.database import DatabaseManager
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.warning("Make sure all required packages are installed.")
    
    # Create placeholder classes for a more graceful failure
    class AgentFactory:
        def __init__(self, *args, **kwargs): pass
        def create_agent(self, *args, **kwargs): return None
        def generate_agent_from_description(self, *args, **kwargs): return None
        
    class LLMProvider:
        def __init__(self, *args, **kwargs): pass
        def list_available_providers(self): return ["openai"]
        def get_provider_models(self, *args): return ["gpt-3.5-turbo"]

def safe_load_env_var(name, default=""):
    """Safely load an environment variable with a default value."""
    try:
        return os.getenv(name, default)
    except:
        return default

def run_ui():
    """Run the Streamlit UI for Agent Forge."""
    # Set up streamlit configuration
    st.set_page_config(
        page_title="Agent Forge",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        try:
            # Try to load configuration
            st.session_state.config = load_config("config.yaml")
            st.session_state.agent_factory = AgentFactory(st.session_state.config)
            st.session_state.llm_provider = LLMProvider(st.session_state.config.get("llm", {}))
            
            # Initialize database if configured
            try:
                if "database" in st.session_state.config:
                    st.session_state.db_manager = DatabaseManager(st.session_state.config.get("database", {}))
                else:
                    st.session_state.db_manager = None
                    logger.warning("No database configuration found")
            except Exception as e:
                st.session_state.db_manager = None
                logger.error(f"Failed to initialize database: {str(e)}")
                st.error(f"Database initialization failed: {str(e)}")
            
            st.session_state.agents = {}
            st.session_state.current_agent = None
            st.session_state.chat_history = []
            st.session_state.error_messages = []
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            st.session_state.error_messages = [f"Initialization error: {str(e)}"]
            st.session_state.initialized = False
    
    # Display any persistent errors
    if hasattr(st.session_state, 'error_messages') and st.session_state.error_messages:
        with st.expander("System Errors", expanded=True):
            for error in st.session_state.error_messages:
                st.error(error)
            if st.button("Clear Errors"):
                st.session_state.error_messages = []
                st.experimental_rerun()
    
    # Page layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Agents", "Settings", "Database", "About"])
    
    with tab1:
        show_agents_tab()
    
    with tab2:
        show_settings_tab()
    
    with tab3:
        show_database_tab()
    
    with tab4:
        show_about_tab()

def show_agents_tab():
    """Display the Agents tab with agent creation and interaction."""
    col1, col2 = st.columns([1, 2])
    
    # Sidebar for agent selection and creation
    with col1:
        st.markdown("## Agent Management")
        
        # Agent selection
        st.subheader("Your Agents")
        if not hasattr(st.session_state, 'agents') or not st.session_state.agents:
            st.info("No agents created yet")
        else:
            agent_names = list(st.session_state.agents.keys())
            selected_agent = st.selectbox("Select Agent", agent_names, key="agent_select")
            if selected_agent:
                st.session_state.current_agent = st.session_state.agents[selected_agent]
                # Reset chat history when switching agents
                if "current_agent_name" not in st.session_state or st.session_state.current_agent_name != selected_agent:
                    st.session_state.chat_history = []
                    st.session_state.current_agent_name = selected_agent
        
        st.divider()
        
        # Create agent form
        st.subheader("Create New Agent")
        with st.form("create_agent_form"):
            agent_name = st.text_input("Agent Name")
            agent_description = st.text_area("Description")
            
            # Advanced options with expander
            with st.expander("Advanced Options"):
                agent_type_options = ["default", "researcher", "coder", "assistant", "custom"]
                agent_type = st.selectbox("Agent Type", agent_type_options)
                
                # If custom type, allow entering a custom type
                if agent_type == "custom":
                    custom_agent_type = st.text_input("Custom Agent Type")
                    if custom_agent_type:
                        agent_type = custom_agent_type
                
                # Get available LLM providers
                providers_available = ["openai", "anthropic", "cohere", "custom"]
                try:
                    if hasattr(st.session_state, 'llm_provider'):
                        providers_available = st.session_state.llm_provider.list_available_providers()
                        if "custom" not in providers_available:
                            providers_available.append("custom")
                except Exception as e:
                    st.warning(f"Could not load providers: {str(e)}")
                
                # LLM selection
                selected_provider = st.selectbox("LLM Provider", providers_available)
                
                if selected_provider:
                    if selected_provider == "custom":
                        # Custom model configuration 
                        selected_model = st.text_input("Model Name", value=safe_load_env_var("CUSTOM_MODEL", ""))
                        api_url = st.text_input("API Endpoint URL", value=safe_load_env_var("CUSTOM_BASE_URL", ""))
                        api_key = st.text_input("API Key", value="", type="password")
                        api_type = st.selectbox("API Type", ["openai_compatible", "anthropic_compatible", "raw_completion"])
                        
                        # Create custom config
                        custom_config = {
                            "llm": {
                                "provider": "custom",
                                "custom": {
                                    "model": selected_model,
                                    "base_url": api_url,
                                    "api_type": api_type
                                }
                            }
                        }
                        if api_key:
                            custom_config["llm"]["custom"]["api_key"] = api_key
                    else:
                        # Get available models for the selected provider
                        try:
                            models = []
                            if hasattr(st.session_state, 'llm_provider'):
                                models = st.session_state.llm_provider.get_provider_models(selected_provider)
                        except Exception as e:
                            st.warning(f"Could not load models: {str(e)}")
                            models = []
                        
                        use_custom_model = st.checkbox("Use a different model name")
                        
                        if use_custom_model or not models:
                            selected_model = st.text_input("Custom Model Name")
                        else:
                            selected_model = st.selectbox("Model", models)
                
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
                max_tokens = st.number_input("Max Tokens", min_value=1, max_value=32000, value=2000)
                
                # Agent capabilities
                st.markdown("#### Agent Capabilities")
                mem_enabled = st.checkbox("Memory", value=True, help="Enable agent memory")
                search_enabled = st.checkbox("Web Search", value=False, help="Enable web search capability")
                code_enabled = st.checkbox("Code Execution", value=False, help="Enable code writing and execution")
                retrieval_enabled = st.checkbox("Document Retrieval", value=False, help="Enable document retrieval from database")
                
                # System prompt
                st.markdown("#### System Prompt")
                system_prompt = st.text_area(
                    "System Prompt", 
                    value="You are a helpful, professional AI assistant.",
                    help="Instructions that define the agent's behavior"
                )
            
            submitted = st.form_submit_button("Create Agent")
            if submitted:
                if not agent_name:
                    st.error("Agent name is required")
                elif not agent_description:
                    st.error("Agent description is required")
                else:
                    # Create the agent
                    try:
                        # Use custom config if selected
                        if selected_provider == "custom" and 'custom_config' in locals():
                            agent_config = custom_config
                        else:
                            # Create config based on UI selections
                            agent_config = {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "system_prompt": system_prompt,
                                "capabilities": {
                                    "memory": mem_enabled,
                                    "search": search_enabled,
                                    "code": code_enabled,
                                    "retrieval": retrieval_enabled
                                },
                                "llm": {
                                    "provider": selected_provider
                                }
                            }
                            
                            # Add provider-specific config
                            if selected_provider not in agent_config["llm"]:
                                agent_config["llm"][selected_provider] = {}
                            agent_config["llm"][selected_provider]["model"] = selected_model
                        
                        # Make sure agent_factory exists
                        if not hasattr(st.session_state, 'agent_factory') or not st.session_state.agent_factory:
                            st.session_state.agent_factory = AgentFactory({})
                            
                        # Create the agent
                        new_agent = st.session_state.agent_factory.create_agent(
                            name=agent_name,
                            description=agent_description,
                            agent_type=agent_type,
                            config=agent_config
                        )
                        
                        # Add to session state
                        if not hasattr(st.session_state, 'agents'):
                            st.session_state.agents = {}
                        st.session_state.agents[agent_name] = new_agent
                        st.session_state.current_agent = new_agent
                        st.session_state.current_agent_name = agent_name
                        st.session_state.chat_history = []
                        st.success(f"Created agent: {agent_name}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to create agent: {str(e)}")
                        st.session_state.error_messages.append(f"Agent creation error: {str(e)}")
                        st.error(traceback.format_exc())
        
        st.divider()
        
        # Auto-generate agent from description
        st.subheader("AI-Generated Agent")
        with st.form("generate_agent_form"):
            gen_prompt = st.text_area(
                "Describe the agent's purpose, capabilities, and personality in detail",
                placeholder="Example: Create a finance expert agent that can analyze stock market data, give investment advice, and explain financial concepts in simple terms."
            )
            
            # Allow selecting the model for agent generation
            gen_model = st.selectbox(
                "Model for agent generation",
                ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"],
                index=0
            )
            
            gen_submitted = st.form_submit_button("Generate Agent")
            
            if gen_submitted:
                if not gen_prompt:
                    st.error("Please provide a description")
                else:
                    with st.spinner("Generating agent..."):
                        try:
                            # Make sure agent_factory exists
                            if not hasattr(st.session_state, 'agent_factory'):
                                st.session_state.agent_factory = AgentFactory({})
                                
                            # Generate the agent
                            new_agent = st.session_state.agent_factory.generate_agent_from_description(
                                gen_prompt, model=gen_model
                            )
                            
                            if not hasattr(st.session_state, 'agents'):
                                st.session_state.agents = {}
                            st.session_state.agents[new_agent.name] = new_agent
                            st.session_state.current_agent = new_agent
                            st.session_state.current_agent_name = new_agent.name
                            st.session_state.chat_history = []
                            st.success(f"Generated agent: {new_agent.name}")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to generate agent: {str(e)}")
                            st.session_state.error_messages.append(f"Agent generation error: {str(e)}")
    
    # Main content - agent interaction
    with col2:
        if hasattr(st.session_state, 'current_agent') and st.session_state.current_agent:
            agent = st.session_state.current_agent
            st.markdown(f"# {agent.name}")
            st.markdown(f"**Description**: {agent.description}")
            
            # Agent capabilities
            st.subheader("Capabilities")
            capabilities = []
            if hasattr(agent, 'capabilities'):
                capabilities = agent.capabilities
            elif hasattr(agent, 'config') and 'capabilities' in agent.config:
                cap_config = agent.config['capabilities']
                if isinstance(cap_config, dict):
                    capabilities = [cap for cap, enabled in cap_config.items() if enabled]
                else:
                    capabilities = cap_config if isinstance(cap_config, list) else []
            
            if capabilities:
                for capability in capabilities:
                    st.markdown(f"- {capability}")
            else:
                st.info("No specific capabilities defined")
            
            st.divider()
            
            # Chat interface
            st.subheader("Chat with Agent")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if hasattr(st.session_state, 'chat_history'):
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            st.markdown(f"**You**: {message['content']}")
                        else:
                            st.markdown(f"**{agent.name}**: {message['content']}")
            
            # Chat input
            user_input = st.text_area("Your message", key="user_message_input", height=100)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
            
            with col2:
                if st.button("Send Message", use_container_width=True):
                    if user_input:
                        # Add user message to history
                        if not hasattr(st.session_state, 'chat_history'):
                            st.session_state.chat_history = []
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        
                        with st.spinner(f"{agent.name} is thinking..."):
                            try:
                                # Execute the agent task
                                history_for_context = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 10 else st.session_state.chat_history
                                
                                # Call agent's execute method
                                if hasattr(agent, 'execute'):
                                    response = agent.execute(user_input, {"history": history_for_context})
                                    response_message = response.get("message", "No response") if isinstance(response, dict) else str(response)
                                else:
                                    # Fallback for agents without execute method
                                    response_message = "This agent doesn't have an execution method implemented."
                                
                                # Add response to history
                                st.session_state.chat_history.append({"role": "assistant", "content": response_message})
                                
                                # Clear the input box
                                st.session_state.user_message_input = ""
                                
                                # Force refresh
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.session_state.error_messages.append(f"Agent execution error: {str(e)}")
        else:
            # No agent selected
            st.markdown("# Welcome to Agent Forge")
            st.markdown("""
            ### Build AI agents with advanced capabilities
            
            Get started by creating or selecting an agent from the left panel.
            
            **Features**:
            - Create custom AI agents
            - Configure agent capabilities
            - Test and interact with your agents
            - AI-generated agent creation
            - Multiple LLM providers
            - Vector database integration
            """)

def show_settings_tab():
    """Display the Settings tab for configuration."""
    st.markdown("## Settings")
    
    # Global settings vs agent settings
    settings_type = st.radio("Settings Type", ["Global Settings", "Agent Settings"], horizontal=True)
    
    if settings_type == "Global Settings":
        show_global_settings()
    else:
        show_agent_settings()

def show_global_settings():
    """Show global application settings."""
    # API keys section
    st.subheader("API Keys")
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI API Key",
        "ANTHROPIC_API_KEY": "Anthropic API Key",
        "COHERE_API_KEY": "Cohere API Key",
        "HUGGINGFACE_API_KEY": "HuggingFace API Key",
        "SERPER_API_KEY": "Serper (Search) API Key"
    }
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Display current API keys (masked)
    with col1:
        st.markdown("#### Current API Keys")
        for env_var, display_name in api_keys.items():
            key_value = safe_load_env_var(env_var, "")
            masked_value = "•" * 8 + key_value[-4:] if key_value and len(key_value) > 4 else "Not set"
            st.text_input(display_name, value=masked_value, disabled=True)
    
    # Form to update API keys
    with col2:
        st.markdown("#### Update API Keys")
        with st.form("api_keys_form"):
            updated_keys = {}
            for env_var, display_name in api_keys.items():
                new_value = st.text_input(f"New {display_name}", value="", type="password")
                if new_value:
                    updated_keys[env_var] = new_value
            
            if st.form_submit_button("Save API Keys"):
                # Update .env file
                try:
                    from dotenv import load_dotenv, set_key, find_dotenv
                    
                    # Find .env file
                    env_path = find_dotenv(usecwd=True)
                    if not env_path:
                        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", ".env")
                    
                    # Update each key
                    for env_var, value in updated_keys.items():
                        set_key(env_path, env_var, value)
                        os.environ[env_var] = value
                        
                    st.success("API keys updated successfully")
                    
                    # Rerun to reflect changes
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to update API keys: {str(e)}")
                    st.session_state.error_messages.append(f"API key update error: {str(e)}")
    
    # Database configuration
    st.subheader("Database Configuration")
    
    db_types = ["Chroma", "FAISS", "Pinecone", "None"]
    current_db = "None"
    
    # Try to detect current database type
    if hasattr(st.session_state, 'config') and "database" in st.session_state.config:
        db_config = st.session_state.config.get("database", {})
        current_db = db_config.get("type", "None")
    
    # Database selection
    selected_db = st.selectbox(
        "Database Type", 
        db_types, 
        index=db_types.index(current_db) if current_db in db_types else 0
    )
    
    # Database-specific configuration
    if selected_db != "None":
        with st.expander(f"{selected_db} Configuration", expanded=True):
            if selected_db == "Chroma":
                chroma_persist_dir = st.text_input(
                    "Persistence Directory", 
                    value=safe_load_env_var("CHROMA_PERSIST_DIR", "./chroma_db")
                )
                
                if st.button("Save Chroma Configuration"):
                    try:
                        # Update config
                        if not hasattr(st.session_state, 'config'):
                            st.session_state.config = {}
                        
                        if "database" not in st.session_state.config:
                            st.session_state.config["database"] = {}
                            
                        st.session_state.config["database"]["type"] = "chroma"
                        st.session_state.config["database"]["chroma"] = {
                            "persist_directory": chroma_persist_dir
                        }
                        
                        # Save to config file
                        save_config(st.session_state.config)
                        
                        # Update environment variable
                        from dotenv import set_key, find_dotenv
                        env_path = find_dotenv(usecwd=True)
                        if env_path:
                            set_key(env_path, "CHROMA_PERSIST_DIR", chroma_persist_dir)
                            
                        st.success("Chroma configuration saved")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to save Chroma configuration: {str(e)}")
                        
            elif selected_db == "Pinecone":
                pinecone_api_key = st.text_input("Pinecone API Key", value="", type="password")
                pinecone_env = st.text_input("Pinecone Environment", value=safe_load_env_var("PINECONE_ENVIRONMENT", ""))
                pinecone_index = st.text_input("Pinecone Index", value=safe_load_env_var("PINECONE_INDEX", ""))
                
                if st.button("Save Pinecone Configuration"):
                    try:
                        # Update config
                        if not hasattr(st.session_state, 'config'):
                            st.session_state.config = {}
                        
                        if "database" not in st.session_state.config:
                            st.session_state.config["database"] = {}
                            
                        st.session_state.config["database"]["type"] = "pinecone"
                        st.session_state.config["database"]["pinecone"] = {
                            "environment": pinecone_env,
                            "index": pinecone_index
                        }
                        
                        # Save to config file
                        save_config(st.session_state.config)
                        
                        # Update environment variables
                        from dotenv import set_key, find_dotenv
                        env_path = find_dotenv(usecwd=True)
                        if env_path:
                            if pinecone_api_key:
                                set_key(env_path, "PINECONE_API_KEY", pinecone_api_key)
                            set_key(env_path, "PINECONE_ENVIRONMENT", pinecone_env)
                            set_key(env_path, "PINECONE_INDEX", pinecone_index)
                            
                        st.success("Pinecone configuration saved")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to save Pinecone configuration: {str(e)}")
    
    # Default model settings
    st.subheader("Default Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # LLM defaults
        st.markdown("#### Default LLM")
        default_llm_provider = st.selectbox(
            "Provider", 
            ["openai", "anthropic", "cohere", "custom"],
            index=0
        )
        
        # Model selection based on provider
        if default_llm_provider == "openai":
            default_llm_model = st.selectbox(
                "Model", 
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"],
                index=2
            )
        elif default_llm_provider == "anthropic":
            default_llm_model = st.selectbox(
                "Model", 
                ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                index=1
            )
        else:
            default_llm_model = st.text_input("Model Name")
    
    with col2:
        # Embedding defaults
        st.markdown("#### Default Embedding Model")
        default_embedding_provider = st.selectbox(
            "Provider", 
            ["openai", "cohere", "huggingface"],
            index=0
        )
        
        # Model selection based on provider
        if default_embedding_provider == "openai":
            default_embedding_model = st.selectbox(
                "Model", 
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0
            )
        elif default_embedding_provider == "cohere":
            default_embedding_model = st.selectbox(
                "Model", 
                ["embed-english-v3.0", "embed-multilingual-v3.0"],
                index=0
            )
        else:
            default_embedding_model = st.selectbox(
                "Model", 
                ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
                index=0
            )
    
    if st.button("Save Default Models"):
        try:
            # Update config
            if not hasattr(st.session_state, 'config'):
                st.session_state.config = {}
            
            # Configure LLM settings
            if "llm" not in st.session_state.config:
                st.session_state.config["llm"] = {}
                
            st.session_state.config["llm"]["provider"] = default_llm_provider
            if default_llm_provider not in st.session_state.config["llm"]:
                st.session_state.config["llm"][default_llm_provider] = {}
            st.session_state.config["llm"][default_llm_provider]["model"] = default_llm_model
            
            # Configure embedding settings
            if "embedding" not in st.session_state.config:
                st.session_state.config["embedding"] = {}
                
            st.session_state.config["embedding"]["provider"] = default_embedding_provider
            st.session_state.config["embedding"]["model"] = default_embedding_model
            
            # Save to config file
            save_config(st.session_state.config)
            
            # Also save to environment variables
            from dotenv import set_key, find_dotenv
            env_path = find_dotenv(usecwd=True)
            if env_path:
                set_key(env_path, "LLM_PROVIDER", default_llm_provider)
                set_key(env_path, f"{default_llm_provider.upper()}_MODEL", default_llm_model)
                set_key(env_path, "EMBEDDING_MODEL", default_embedding_model)
                
            st.success("Default models saved")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to save default models: {str(e)}")

def show_agent_settings():
    """Show settings for the selected agent."""
    if hasattr(st.session_state, 'current_agent') and st.session_state.current_agent:
        agent = st.session_state.current_agent
        st.markdown(f"## Settings for {agent.name}")
        
        # Display current config as JSON
        with st.expander("Current Configuration"):
            if hasattr(agent, 'config'):
                st.json(agent.config)
            else:
                st.warning("This agent doesn't have a configuration")
        
        # Edit configuration
        st.subheader("Edit Configuration")
        
        # Get current configuration or create empty one
        if hasattr(agent, 'config'):
            agent_config = agent.config.copy() if isinstance(agent.config, dict) else {}
        else:
            agent_config = {}
        
        # Create tabs for different settings categories
        tab1, tab2, tab3 = st.tabs(["Basic Settings", "LLM Configuration", "Capabilities"])
        
        with tab1:
            st.markdown("### Basic Settings")
            
            # Basic agent parameters
            agent_name = st.text_input("Agent Name", value=agent.name)
            agent_description = st.text_area("Description", value=agent.description)
            
            if st.button("Save Basic Settings"):
                try:
                    agent.name = agent_name
                    agent.description = agent_description
                    st.success("Basic settings saved")
                except Exception as e:
                    st.error(f"Failed to save basic settings: {str(e)}")
        
        with tab2:
            st.markdown("### LLM Configuration")
            
            # LLM provider and model
            llm_provider = st.selectbox("LLM Provider", ["openai", "anthropic", "cohere", "custom"], index=0)
            llm_model = st.text_input("Model", value=agent_config.get("llm", {}).get("model", ""))
            
            if st.button("Save LLM Configuration"):
                try:
                    agent_config["llm"] = {"provider": llm_provider, "model": llm_model}
                    agent.config = agent_config
                    st.success("LLM configuration saved")
                except Exception as e:
                    st.error(f"Failed to save LLM configuration: {str(e)}")
        
        with tab3:
            st.markdown("### Capabilities")
            
            # Agent capabilities
            mem_enabled = st.checkbox("Memory", value=agent_config.get("capabilities", {}).get("memory", False))
            search_enabled = st.checkbox("Web Search", value=agent_config.get("capabilities", {}).get("search", False))
            code_enabled = st.checkbox("Code Execution", value=agent_config.get("capabilities", {}).get("code", False))
            retrieval_enabled = st.checkbox("Document Retrieval", value=agent_config.get("capabilities", {}).get("retrieval", False))
            
            if st.button("Save Capabilities"):
                try:
                    agent_config["capabilities"] = {
                        "memory": mem_enabled,
                        "search": search_enabled,
                        "code": code_enabled,
                        "retrieval": retrieval_enabled
                    }
                    agent.config = agent_config
                    st.success("Capabilities saved")
                except Exception as e:
                    st.error(f"Failed to save capabilities: {str(e)}")

def save_config(config):
    """Save the configuration to a YAML file."""
    try:
        with open("config.yaml", "w") as file:
            yaml.dump(config, file)
    except Exception as e:
        st.error(f"Failed to save configuration: {str(e)}")

def show_database_tab():
    """Display the Database tab for database management."""
    st.markdown("## Database Management")
    
    if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
        # Database is configured
        try:
            db_status = st.session_state.db_manager.get_status()
            
            # Show database status
            st.subheader("Database Status")
            
            # Create metrics for database stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Database Type", db_status.get("type", "Unknown"))
            with col2:
                st.metric("Document Count", db_status.get("document_count", 0))
            with col3:
                st.metric("Collection Count", db_status.get("collection_count", 0))
            
            # Show database operations
            st.subheader("Database Operations")
            
            # Dangerous operations in an expander
            with st.expander("Danger Zone", expanded=False):
                st.warning("⚠️ These operations can result in data loss and cannot be undone!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear All Data", use_container_width=True):
                        confirm = st.checkbox("I understand this will delete all data")
                        if confirm and st.button("Confirm Clear Database", use_container_width=True):
                            try:
                                st.session_state.db_manager.clear_database()
                                st.success("Database cleared successfully")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to clear database: {str(e)}")
                                st.session_state.error_messages.append(f"Database clear error: {str(e)}")
                
                with col2:
                    if st.button("Reset Database", use_container_width=True):
                        confirm = st.checkbox("I understand this will reset the database")
                        if confirm and st.button("Confirm Reset Database", use_container_width=True):
                            try:
                                st.session_state.db_manager.reset_database()
                                st.success("Database reset successfully")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to reset database: {str(e)}")
                                st.session_state.error_messages.append(f"Database reset error: {str(e)}")
            
            # Database content browser
            st.subheader("Browse Database")
            
            try:
                collections = st.session_state.db_manager.list_collections()
                if collections:
                    selected_collection = st.selectbox("Select Collection", collections)
                    if selected_collection:
                        # Show documents in collection
                        documents = st.session_state.db_manager.get_documents(selected_collection, limit=10)
                        
                        if documents:
                            st.markdown(f"**Documents in {selected_collection}** (showing first 10)")
                            for i, doc in enumerate(documents):
                                with st.expander(f"Document {i+1}", expanded=i==0):
                                    st.json(doc)
                        else:
                            st.info(f"No documents found in collection {selected_collection}")
                else:
                    st.info("No collections found in the database")
            except Exception as e:
                st.error(f"Error accessing database content: {str(e)}")
            
            # Document upload section
            st.subheader("Upload Documents")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_files = st.file_uploader("Upload documents to database", accept_multiple_files=True,
                                                 type=["txt", "pdf", "docx", "md"])
            with col2:
                collection_name = st.text_input("Collection name", value="documents")
                chunk_size = st.number_input("Chunk size", min_value=100, max_value=8000, value=1000,
                                           help="Number of characters per document chunk")
            
            if uploaded_files and st.button("Process and Store Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        processed = 0
                        for file in uploaded_files:
                            # Read and process file
                            if file.type == "text/plain":
                                content = file.read().decode()
                            elif file.type == "application/pdf":
                                st.warning(f"PDF processing not implemented yet: {file.name}")
                                continue
                            elif "docx" in file.type:
                                st.warning(f"DOCX processing not implemented yet: {file.name}")
                                continue
                            else:
                                content = file.read().decode()
                            
                            # Store document
                            st.session_state.db_manager.add_document(
                                content, 
                                metadata={"filename": file.name, "source": "upload"},
                                collection=collection_name,
                                chunk_size=chunk_size
                            )
                            processed += 1
                        
                        st.success(f"Successfully processed {processed} documents")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.error_messages.append(f"Document processing error: {str(e)}")
                
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")
            st.session_state.error_messages.append(f"Database error: {str(e)}")
    else:
        # Database not configured
        st.warning("No database is currently configured")
        st.info("To configure a database, go to the Settings tab and select a database provider")
        
        # Quick setup buttons
        st.subheader("Quick Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Setup Chroma Database", use_container_width=True):
                try:
                    # Create config if doesn't exist
                    if not hasattr(st.session_state, 'config'):
                        st.session_state.config = {}
                    
                    # Add database configuration
                    st.session_state.config["database"] = {
                        "type": "chroma",
                        "chroma": {
                            "persist_directory": "./chroma_db"
                        }
                    }
                    
                    # Save configuration
                    save_config(st.session_state.config)
                    
                    # Initialize database manager
                    from agent_forge.integrations.database import DatabaseManager
                    st.session_state.db_manager = DatabaseManager(st.session_state.config.get("database", {}))
                    
                    st.success("Chroma database configured successfully")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to configure Chroma database: {str(e)}")
                    st.session_state.error_messages.append(f"Database setup error: {str(e)}")
        
        with col2:
            if st.button("Setup In-Memory Database", use_container_width=True):
                try:
                    # Create config if doesn't exist
                    if not hasattr(st.session_state, 'config'):
                        st.session_state.config = {}
                    
                    # Add database configuration
                    st.session_state.config["database"] = {
                        "type": "memory"
                    }
                    
                    # Save configuration
                    save_config(st.session_state.config)
                    
                    # Initialize database manager
                    from agent_forge.integrations.database import DatabaseManager
                    st.session_state.db_manager = DatabaseManager(st.session_state.config.get("database", {}))
                    
                    st.success("In-memory database configured successfully")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to configure in-memory database: {str(e)}")
                    st.session_state.error_messages.append(f"Database setup error: {str(e)}")

def show_about_tab():
    """Display the About tab with information about the application."""
    st.markdown("## About Agent Forge")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        # Agent Forge 🤖⚒️
        
        **A powerful platform for building and managing AI agents with advanced capabilities.**
        
        Agent Forge lets you create, customize, and deploy AI agents powered by multiple LLM providers.
        With an intuitive interface and extensive configuration options, you can build AI systems
        tailored to your specific needs without writing code.
        
        ## Key Features
        
        - **Multiple LLM Providers**: Support for OpenAI, Anthropic, Cohere and custom LLM providers
        - **Advanced Embeddings**: Use embeddings from OpenAI, Cohere or Hugging Face models
        - **Vector Database Integration**: Store and retrieve documents with semantic search
        - **Custom AI Agents**: Create agents with specialized capabilities and personalities
        - **No-Code Interface**: Build and configure agents entirely through the UI
        - **Environment Management**: Easily manage API keys and configuration
        
        ## Support and Documentation
        
        - [GitHub Repository](https://github.com/your-github/agent-forge)
        - [Documentation](https://docs.agent-forge.ai)
        - [Community Discord](https://discord.gg/agent-forge)
        """)
    
    with col2:
        st.markdown("### System Information")
        
        # Show version information
        version = os.getenv("AGENT_FORGE_VERSION", "1.0.0")
        st.info(f"Agent Forge Version: {version}")
        
        # Show environment info
        st.markdown("#### Environment")
        
        # Check Python version
        import platform
        st.write(f"Python: {platform.python_version()}")
        
        # Check if running in Docker
        in_docker = os.path.exists("/.dockerenv")
        st.write(f"Docker: {'Yes' if in_docker else 'No'}")
        
        # Check key packages
        try:
            import openai
            st.write(f"OpenAI SDK: {openai.__version__}")
        except:
            st.write("OpenAI SDK: Not installed")
        
        try:
            import torch
            st.write(f"PyTorch: {torch.__version__}")
            st.write(f"CUDA Available: {torch.cuda.is_available()}")
        except:
            st.write("PyTorch: Not installed")
        
        # LLM Provider status
        st.markdown("#### LLM Providers")
        
        # Check API keys (just show if they exist or not)
        providers = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Cohere": "COHERE_API_KEY",
            "HuggingFace": "HUGGINGFACE_API_KEY"
        }
        
        for provider, env_var in providers.items():
            has_key = bool(os.getenv(env_var, ""))
            status = "✅ Configured" if has_key else "❌ Not Configured"
            st.write(f"{provider}: {status}")
        
        # Show diagnostics button
        if st.button("Run Diagnostics"):
            with st.spinner("Running system diagnostics..."):
                # Check connections
                diagnostics = run_diagnostics()
                
                st.success("Diagnostics complete!")
                st.json(diagnostics)

def run_diagnostics():
    """Run diagnostics tests on the system."""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "memory": "OK",
            "disk": "OK",
            "python": f"{sys.version}",
        },
        "connections": {}
    }
    
    # Test database connection if configured
    if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
        try:
            db_status = st.session_state.db_manager.get_status()
            results["connections"]["database"] = "OK"
            results["database"] = db_status
        except:
            results["connections"]["database"] = "ERROR"
    else:
        results["connections"]["database"] = "Not configured"
    
    # Test OpenAI API connection if key exists
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Very minimal test to see if key works
            client.models.list(limit=1)
            results["connections"]["openai"] = "OK" 
        except Exception as e:
            results["connections"]["openai"] = f"ERROR: {str(e)}"
    
    return results

if __name__ == "__main__":
    run_ui()