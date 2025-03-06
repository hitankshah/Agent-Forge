import streamlit as st
import os
import json
from typing import Dict, Any, List, Optional
import time

from .agent_builder import AgentBuilder, AgentSpecification
from .model_providers import ModelProvider, OpenAIProvider, AnthropicProvider

class AgentBuilderUI:
    """UI class for Agent Builder using Streamlit"""
    
    def __init__(self):
        # Initialize session state variables if they don't exist
        if 'agent_specs' not in st.session_state:
            st.session_state.agent_specs = {}
        if 'current_agent' not in st.session_state:
            st.session_state.current_agent = None
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {
                'openai': os.environ.get('OPENAI_API_KEY', ''),
                'anthropic': os.environ.get('ANTHROPIC_API_KEY', '')
            }
        
        # Create builder instance
        self.builder = self._create_builder()
        
        # Paths for saving/loading data
        self.agents_dir = os.path.join(os.path.dirname(__file__), '..', 'agents')
        os.makedirs(self.agents_dir, exist_ok=True)
        
    def _create_builder(self) -> AgentBuilder:
        """Create AgentBuilder with current API keys"""
        return AgentBuilder(
            openai_api_key=st.session_state.api_keys.get('openai'),
            anthropic_api_key=st.session_state.api_keys.get('anthropic')
        )
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Agent Forge")
        
        menu_options = ["Build Agent", "My Agents", "Settings"]
        selection = st.sidebar.radio("Navigation", menu_options)
        
        st.sidebar.divider()
        
        # Show available models
        st.sidebar.subheader("Available Models")
        for provider_name, provider in self.builder.model_registry.providers.items():
            with st.sidebar.expander(f"{provider_name.capitalize()} Models"):
                models = provider.get_available_models()
                if models:
                    for model in models[:5]:  # Limit to first 5 models to avoid clutter
                        st.sidebar.text(model)
                else:
                    st.sidebar.text("No models available or API key missing")
        
        return selection
    
    def render_build_agent_page(self):
        """Render the page for building new agents"""
        st.title("Build a New Agent")
        
        # Agent Description Input
        st.subheader("Describe Your Agent")
        agent_description = st.text_area(
            "What would you like your agent to do?",
            height=150,
            placeholder="Example: I need an AI agent that can help researchers analyze scientific papers. " +
                      "It should extract key findings, compare methodologies, and suggest related research."
        )
        
        col1, col2 = st.columns(2)
        analyze_clicked = col1.button("Analyze Requirements", type="primary")
        clear_clicked = col2.button("Clear", type="secondary")
        
        if clear_clicked:
            st.session_state.current_agent = None
            st.experimental_rerun()
            
        if analyze_clicked and agent_description:
            with st.spinner("Analyzing requirements..."):
                # Analyze requirements
                requirements = self.builder.analyze_requirements(agent_description)
                
                if "error" in requirements:
                    st.error(f"Error analyzing requirements: {requirements['error']}")
                else:
                    # Create specification
                    spec = self.builder.create_agent_specification(requirements)
                    st.session_state.current_agent = spec
                    st.success("Analysis complete!")
                    
        # Show agent details if we have a current agent
        if st.session_state.current_agent:
            self._render_agent_details(st.session_state.current_agent)
    
    def _render_agent_details(self, spec: AgentSpecification):
        """Render the details of an analyzed agent"""
        st.divider()
        st.subheader(f"Agent: {spec.name}")
        st.write(f"**Purpose:** {spec.purpose}")
        st.write(f"**Description:** {spec.description}")
        
        # Show capabilities
        st.subheader("Capabilities")
        for cap in spec.capabilities:
            with st.expander(f"{cap.name}"):
                st.write(f"**Description:** {cap.description}")
                st.write(f"**Dependencies:** {', '.join(cap.dependencies)}")
                st.write("**Parameters:**")
                for param_name, param_value in cap.parameters.items():
                    st.write(f"- {param_name}: {param_value}")
                
                # Allow model selection for each capability
                if cap.performance_by_model:
                    st.write("**Model Performance:**")
                    for model, score in cap.performance_by_model.items():
                        st.progress(score, text=f"{model}: {score:.2f}")
        
        # Show knowledge domains
        st.subheader("Knowledge Domains")
        st.write(", ".join(spec.knowledge_domains))
        
        # Show recommended models
        st.subheader("Recommended Models")
        if spec.recommended_models:
            for model in spec.recommended_models:
                st.write(f"- {model}")
        else:
            st.write("No specific models recommended.")
        
        # Implementation controls
        col1, col2, col3 = st.columns(3)
        
        if col1.button("Generate Implementation Plan", key="impl_plan"):
            with st.spinner("Generating implementation plan..."):
                plan = self.builder.generate_implementation_plan(spec)
                st.session_state.implementation_plan = plan
                st.success("Implementation plan generated!")
                st.experimental_rerun()
                
        if col2.button("Generate Code Skeleton", key="code_skel"):
            if hasattr(st.session_state, 'implementation_plan'):
                with st.spinner("Generating code skeletons..."):
                    code = self.builder.generate_code_skeleton(st.session_state.implementation_plan)
                    st.session_state.code_skeletons = code
                    st.success("Code skeletons generated!")
                    st.experimental_rerun()
            else:
                st.error("Please generate an implementation plan first.")
                
        if col3.button("Save Agent", key="save_agent"):
            agent_id = spec.name.lower().replace(" ", "_")
            st.session_state.agent_specs[agent_id] = spec
            self._save_agent(agent_id, spec)
            st.success(f"Agent '{spec.name}' saved!")
            
        # Show implementation plan if available
        if hasattr(st.session_state, 'implementation_plan'):
            st.divider()
            st.subheader("Implementation Plan")
            with st.expander("View Implementation Plan"):
                st.json(st.session_state.implementation_plan)
        
        # Show code skeletons if available
        if hasattr(st.session_state, 'code_skeletons'):
            st.divider()
            st.subheader("Code Skeletons")
            for filename, code in st.session_state.code_skeletons.items():
                with st.expander(f"{filename}"):
                    st.code(code, language="python")
    
    def render_my_agents_page(self):
        """Render page showing existing agents"""
        st.title("My Agents")
        
        # Load agents from files
        self._load_saved_agents()
        
        if not st.session_state.agent_specs:
            st.info("You don't have any saved agents yet. Go to 'Build Agent' to create one!")
            return
            
        # Select agent to view
        agent_names = {agent_id: spec.name for agent_id, spec in st.session_state.agent_specs.items()}
        selected_agent_id = st.selectbox("Select Agent", options=list(agent_names.keys()),
                                       format_func=lambda x: agent_names[x])
        
        if selected_agent_id:
            spec = st.session_state.agent_specs[selected_agent_id]
            self._render_agent_details(spec)
            
            # Additional actions for existing agents
            col1, col2 = st.columns(2)
            
            if col1.button("Evaluate Agent", key="eval_agent"):
                with st.spinner("Evaluating agent design..."):
                    evaluation = self.builder.evaluate_agent_design(spec)
                    st.session_state.current_evaluation = evaluation
                    st.success("Evaluation complete!")
                    st.experimental_rerun()
                    
            if col2.button("Optimize Model Selection", key="opt_model"):
                with st.spinner("Optimizing model selection..."):
                    model_assignments = self.builder.optimize_model_selection(spec)
                    st.session_state.model_assignments = model_assignments
                    st.success("Optimization complete!")
                    st.experimental_rerun()
            
            # Show evaluation if available
            if hasattr(st.session_state, 'current_evaluation'):
                st.divider()
                st.subheader("Agent Evaluation")
                eval_data = st.session_state.current_evaluation
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Completeness", f"{eval_data['completeness']:.2f}")
                col2.metric("Coherence", f"{eval_data['coherence']:.2f}")
                col3.metric("Feasibility", f"{eval_data['feasibility']:.2f}")
                col4.metric("Model Fit", f"{eval_data['model_fit']:.2f}")
                
                st.progress(eval_data["overall_score"], text=f"Overall Score: {eval_data['overall_score']:.2f}")
                
                if eval_data["issues"]:
                    st.subheader("Issues")
                    for issue in eval_data["issues"]:
                        st.warning(issue)
                
                if eval_data["recommendations"]:
                    st.subheader("Recommendations")
                    for rec in eval_data["recommendations"]:
                        st.info(rec)
            
            # Show model assignments if available
            if hasattr(st.session_state, 'model_assignments'):
                st.divider()
                st.subheader("Optimized Model Selection")
                
                model_assignments = st.session_state.model_assignments
                for cap_name, model in model_assignments.items():
                    st.write(f"**{cap_name}**: {model}")
    
    def render_settings_page(self):
        """Render the settings page"""
        st.title("Settings")
        
        # API Keys
        st.subheader("API Keys")
        
        # OpenAI API Key
        openai_key = st.text_input("OpenAI API Key", 
                                  value=st.session_state.api_keys.get('openai', ''),
                                  type="password")
        
        # Anthropic API Key
        anthropic_key = st.text_input("Anthropic API Key", 
                                     value=st.session_state.api_keys.get('anthropic', ''),
                                     type="password")
        
        if st.button("Save API Keys"):
            st.session_state.api_keys['openai'] = openai_key
            st.session_state.api_keys['anthropic'] = anthropic_key
            
            # Recreate the builder with new API keys
            self.builder = self._create_builder()
            st.success("API keys saved!")
        
        # Model Settings
        st.subheader("Default Models")
        
        # Get available models
        openai_models = []
        anthropic_models = []
        
        if openai_key:
            openai_provider = self.builder.model_registry.providers.get('openai')
            if openai_provider:
                openai_models = openai_provider.get_available_models()
        
        if anthropic_key:
            anthropic_provider = self.builder.model_registry.providers.get('anthropic')
            if anthropic_provider:
                anthropic_models = anthropic_provider.get_available_models()
        
        # Default OpenAI Model
        if openai_models:
            default_openai = st.selectbox("Default OpenAI Model", 
                                        options=openai_models,
                                        index=0 if 'gpt-4' not in openai_models else openai_models.index('gpt-4'))
            
            if st.button("Set Default OpenAI Model"):
                openai_provider = self.builder.model_registry.providers.get('openai')
                if openai_provider:
                    openai_provider.default_model = default_openai
                    st.success(f"Default OpenAI model set to {default_openai}")
        
        # Default Anthropic Model
        if anthropic_models:
            default_anthropic = st.selectbox("Default Anthropic Model", 
                                           options=anthropic_models,
                                           index=0)
            
            if st.button("Set Default Anthropic Model"):
                anthropic_provider = self.builder.model_registry.providers.get('anthropic')
                if anthropic_provider:
                    anthropic_provider.default_model = default_anthropic
                    st.success(f"Default Anthropic model set to {default_anthropic}")
    
    def _save_agent(self, agent_id: str, spec: AgentSpecification):
        """Save agent to file"""
        filepath = os.path.join(self.agents_dir, f"{agent_id}.json")
        self.builder.save_agent_specification(spec, filepath)
    
    def _load_saved_agents(self):
        """Load saved agents from files"""
        if os.path.exists(self.agents_dir):
            for filename in os.listdir(self.agents_dir):
                if filename.endswith('.json'):
                    agent_id = filename[:-5]  # Remove .json
                    if agent_id not in st.session_state.agent_specs:
                        filepath = os.path.join(self.agents_dir, filename)
                        spec = self.builder.load_agent_specification(filepath)
                        if spec:
                            st.session_state.agent_specs[agent_id] = spec
    
    def run(self):
        """Run the UI application"""
        page_selection = self.render_sidebar()
        
        if page_selection == "Build Agent":
            self.render_build_agent_page()
        elif page_selection == "My Agents":
            self.render_my_agents_page()
        elif page_selection == "Settings":
            self.render_settings_page()
