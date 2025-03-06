"""
Integrated UI that combines agent_builder and agent_forge components
into a single interface with enhanced capabilities.
"""

import streamlit as st
# Must call set_page_config as the first Streamlit command
st.set_page_config(
    page_title="AI Agent Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import random

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegratedUI")

# Define variables to store imported components
UnifiedPlatform = None
render_agent_metrics_chart = None
render_capability_network = None
apply_theme = None
get_available_themes = None

# Import components without using Streamlit commands
try:
    from unified_platform import UnifiedPlatform
    from components.visualizations import render_agent_metrics_chart, render_capability_network
    from theme import apply_theme, get_available_themes
except ImportError:
    logger.error("Failed to import required modules. Make sure all dependencies are installed.")
    # Don't use st.error here as it would be before set_page_config

def generate_logo(save_path=None):
    """Generate a simple logo if none exists"""
    try:
        # Create a blank image with a gradient background
        width, height = 400, 200
        image = Image.new('RGB', (width, height), color=(25, 25, 35))
        
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Create gradient background
        for y in range(height):
            r = int(25 + (35 - 25) * y / height)
            g = int(25 + (65 - 25) * y / height)
            b = int(35 + (95 - 35) * y / height)
            for x in range(width):
                draw.point((x, y), fill=(r, g, b))
        
        # Draw a network of connected nodes to represent AI agents
        colors = [(240, 84, 84), (77, 175, 124), (66, 139, 202), (255, 193, 7)]
        
        nodes = []
        for _ in range(8):  # Number of nodes
            x = random.randint(30, width - 30)
            y = random.randint(30, height - 30)
            color = random.choice(colors)
            nodes.append((x, y, color))
        
        # Draw connections between nodes
        for i, (x1, y1, _) in enumerate(nodes):
            for j, (x2, y2, _) in enumerate(nodes):
                if i < j and random.random() < 0.5:
                    # Calculate distance to decide on drawing connection
                    dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    if dist < 150:  # Only connect nearby nodes
                        # Draw a semi-transparent connection line
                        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 128), width=1)
        
        # Draw nodes
        for x, y, color in nodes:
            radius = random.randint(5, 12)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((width//2-90, height//2-20), "Agent Platform", font=font, fill=(255, 255, 255))
        
        # Save the logo if requested
        if save_path:
            image.save(save_path)
            
        return image
    except Exception as e:
        logger.error(f"Error generating logo: {e}")
        return None

def get_base64_encoded_image(image_path):
    """Get base64 encoded image for embedding in CSS"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def apply_custom_theme():
    """Apply a custom theme to the Streamlit app"""
    # Generate or get logo
    logo_path = "logo.png"
    if not os.path.exists(logo_path):
        logo_image = generate_logo(logo_path)
    
    # Custom CSS for better styling
    custom_css = """
    <style>
    /* Modern styling */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Dashboard cards */
    div.dashboard-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-top: 4px solid #4b6fff;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div.dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    /* Agent cards */
    div.agent-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4b6fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        transition: all 0.2s ease;
    }
    
    div.agent-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Capability items */
    div.capability-item {
        background-color: #f8f9fb;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #6e8dfb;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a2942 0%, #141e30 100%);
    }
    
    /* Metric improvements */
    div.css-12w0qpk.e1tzin5v2 {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Typography improvements */
    h1 {
        font-weight: 600 !important;
        color: #1a2942 !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        font-weight: 500 !important;
        color: #2c3e50 !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-weight: 500 !important;
        color: #34495e !important;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 6px;
        padding: 0.3rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4b6fff 0%, #6e8dfb 100%);
        border-radius: 10px;
    }
    
    /* Status indicators */
    .status-active {
        color: #2ecc71;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* Input elements */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 6px;
        border-color: #d1d9e6;
    }
    
    /* Make the sidebar navigation look more like a modern app */
    .nav-button {
        border-radius: 6px;
        margin-bottom: 0.3rem;
        transition: all 0.2s;
    }
    
    .nav-button-active {
        background-color: #4b6fff !important;
        color: white !important;
    }
    
    /* Add gradient to expandable sections */
    .st-ae {
        border-radius: 6px;
    }
    
    /* Add animation to chat messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.3s ease-out forwards;
    }
    
    /* Cards for dashboard metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-card h3 {
        margin-top: 0;
        font-size: 1rem;
        color: #6c757d;
    }
    
    .metric-card p {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
        color: #1a2942;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

class IntegratedUI:
    """Integrated UI for agent_builder and agent_forge."""
    
    def __init__(self):
        """Initialize the integrated UI."""
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.current_page = "Home"
            st.session_state.current_agent = None
            st.session_state.api_keys = {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "cohere": os.environ.get("COHERE_API_KEY", "")
            }
            st.session_state.theme = "dark"
        
        # Initialize platform if not already done
        if not st.session_state.initialized:
            self._initialize_platform()
            st.session_state.initialized = True
    
    def _initialize_platform(self):
        """Initialize the unified platform."""
        try:
            # Look for config file
            config_path = os.path.join(os.path.dirname(__file__), "config", "integration_config.json")
            if not os.path.exists(config_path):
                config_path = None
            
            # Initialize platform
            st.session_state.platform = UnifiedPlatform(config_path)
            logger.info("Unified platform initialized")
            
            # Update API keys in session state if they exist in platform config
            if hasattr(st.session_state.platform, "config") and "api_keys" in st.session_state.platform.config:
                for key, value in st.session_state.platform.config["api_keys"].items():
                    if value:
                        st.session_state.api_keys[key] = value
        except Exception as e:
            logger.error(f"Error initializing platform: {e}")
            st.error(f"Error initializing platform: {e}")
    
    def render(self):
        """Render the main UI."""
        # Apply custom theme first
        apply_custom_theme()
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content
        current_page = st.session_state.current_page
        if current_page == "Home":
            self._render_home_page()
        elif current_page == "Build Agent":
            self._render_build_page()
        elif current_page == "My Agents":
            self._render_agents_page()
        elif current_page == "Deploy":
            self._render_deploy_page()
        elif current_page == "Run Agent":
            self._render_run_page()
        elif current_page == "Settings":
            self._render_settings_page()
        elif current_page == "Status":
            self._render_status_page()
    
    def _render_sidebar(self):
        """Render the sidebar navigation."""
        with st.sidebar:
            # Check if logo.png exists
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                # Fix: Replace use_container_width with width parameter
                st.image(logo_path, width=None)
            else:
                logo_image = generate_logo(logo_path)
                if logo_image:
                    # Convert the PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    logo_image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Display the image with proper width parameter
                    st.image(img_byte_arr, width=None)
                else:
                    st.warning("Unable to generate logo. Please create a 'logo.png' file.")
            
            st.title("🤖 Agent Platform")
            st.divider()
            
            # Navigation with improved styling
            pages = [
                ("Home", "🏠"),
                ("Build Agent", "🛠️"),
                ("My Agents", "🤖"),
                ("Deploy", "🚀"),
                ("Run Agent", "💬"),
                ("Settings", "⚙️"),
                ("Status", "📊")
            ]
            
            for page_name, icon in pages:
                button_style = "nav-button nav-button-active" if st.session_state.current_page == page_name else "nav-button"
                if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    try:
                        # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                        st.rerun()
                    except AttributeError:
                        # Fall back to experimental_rerun for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except:
                            st.error("Failed to rerun the app. Please refresh the page manually.")
            
            # Show selected agent info if available
            if st.session_state.current_agent:
                st.divider()
                st.subheader("Selected Agent")
                
                # Create a card-like container for the agent info
                agent = st.session_state.current_agent
                
                # Use HTML for better styling of the agent card
                agent_card_html = f"""
                <div class="agent-card">
                    <h3 style="margin-top:0;">{agent['name']}</h3>
                    <p style="font-style:italic; color:#666;">{agent['description']}</p>
                </div>
                """
                st.markdown(agent_card_html, unsafe_allow_html=True)
                
                if st.button("Clear Selection", key="clear_agent"):
                    st.session_state.current_agent = None
                    try:
                        # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                        st.rerun()
                    except AttributeError:
                        # Fall back to experimental_rerun for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except:
                            st.error("Failed to rerun the app. Please refresh the page manually.")
            
            # Footer
            st.divider()
            status = st.session_state.platform.get_status() if hasattr(st.session_state, 'platform') else {"registry": {"agents_count": 0, "deployments_count": 0}}
            
            # Use a more modern footer with metrics
            footer_html = f"""
            <div style="padding: 0.5rem; border-radius: 5px; font-size: 0.8rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                    <span>Platform v1.0.0</span>
                    <span style="color:#4b6fff;">✓ Online</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: #666;">
                    <span>Agents: {status['registry']['agents_count']}</span>
                    <span>Deployments: {status['registry']['deployments_count']}</span>
                </div>
            </div>
            """
            st.markdown(footer_html, unsafe_allow_html=True)
    
    def _render_home_page(self):
        """Render the home page."""
        st.title("🚀 Unified Agent Platform")
        st.write("Design, build, and deploy AI agents with advanced capabilities")
        
        # Overview metrics with improved styling
        status = st.session_state.platform.get_status() if hasattr(st.session_state, 'platform') else {"registry": {"agents_count": 0, "capabilities_count": 0, "deployments_count": 0}}
        
        # Use markdown for better styled metrics
        metrics_html = """
        <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
        """
        
        # Add metrics to the HTML
        metrics = [
            {"label": "Agents", "value": status["registry"]["agents_count"], "icon": "🤖"},
            {"label": "Capabilities", "value": status["registry"].get("capabilities_count", 0), "icon": "⚡"},
            {"label": "Deployments", "value": status["registry"]["deployments_count"], "icon": "🚀"},
        ]
        
        for metric in metrics:
            metrics_html += f"""
            <div class="dashboard-card" style="flex: 1; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{metric['icon']}</div>
                <h3 style="margin:0; font-size: 0.9rem; color: #6c757d;">{metric['label']}</h3>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0; color: #1a2942;">{metric['value']}</p>
            </div>
            """
        
        metrics_html += "</div>"
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick actions with better styling
        st.subheader("Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use custom HTML for card styling
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="margin-top:0;">Create New Agent</h3>
                <p>Design a custom AI agent with specific capabilities and deploy it to your environment.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Create Agent", key="home_create"):
                st.session_state.current_page = "Build Agent"
                try:
                    # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                    st.rerun()
                except AttributeError:
                    # Fall back to experimental_rerun for older Streamlit versions
                    try:
                        st.experimental_rerun()
                    except:
                        st.error("Failed to rerun the app. Please refresh the page manually.")
        
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="margin-top:0;">Run Existing Agent</h3>
                <p>Launch and interact with your deployed agents to accomplish tasks.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Run Agent", key="home_run"):
                st.session_state.current_page = "Run Agent"
                try:
                    # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                    st.rerun()
                except AttributeError:
                    # Fall back to experimental_rerun for older Streamlit versions
                    try:
                        st.experimental_rerun()
                    except:
                        st.error("Failed to rerun the app. Please refresh the page manually.")

    def _reset_build_wizard(self):
        """Reset the build wizard state to step 1."""
        if 'build_step' in st.session_state:
            st.session_state.build_step = 1
            st.session_state.agent_details = {
                "description": "",
                "name": "",
                "model_provider": "Free Models",
                "model": "Gemma-7b",
                "capabilities": [],
                "knowledge_domains": [],
                "persona": "Professional",
                "template": None,
                "response_length": 3,
                "temperature": 0.7,
                "memory_enabled": True
            }

    def _render_build_page(self):
        """Render the enhanced build agent page with professional UI."""
        st.title("🔧 Build Your AI Agent")
        
        # Initialize the build wizard state if not already done
        if 'build_step' not in st.session_state:
            st.session_state.build_step = 1
            st.session_state.agent_details = {
                "description": "",
                "name": "",
                "model_provider": "Free Models",
                "model": "Gemma-7b",
                "capabilities": [],
                "knowledge_domains": [],
                "persona": "Professional",
                "template": None,
                "response_length": 3,
                "temperature": 0.7,
                "memory_enabled": True
            }
        
        # Render step progress indicator
        steps = ["Agent Type", "Capabilities", "Model Selection", "Settings", "Review"]
        step_status = ["step-complete" if i < st.session_state.build_step else 
                    "step-active" if i == st.session_state.build_step else "" 
                    for i in range(1, 6)]
        
        progress_html = """
        <div class="step-progress">
        """
        
        for i, (step, status) in enumerate(zip(steps, step_status)):
            progress_html += f"""
            <div class="step-item {status}">
                <div class="step-number">{i+1}</div>
                <div class="step-label">{step}</div>
            </div>
            """
        
        progress_html += "</div>"
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Handle different steps of the wizard
        if st.session_state.build_step == 1:
            # Step 1: Choose agent type or template
            self._render_agent_type_step()
        elif st.session_state.build_step == 2:
            # Step 2: Configure capabilities
            self._render_capabilities_step()
        elif st.session_state.build_step == 3:
            # Step 3: Select model
            self._render_model_selection_step()
        elif st.session_state.build_step == 4:
            # Step 4: Additional settings
            self._render_agent_settings_step()
        elif st.session_state.build_step == 5:
            # Step 5: Review and create
            self._render_review_step()

    def _render_agent_settings_step(self):
        """Render the fourth step of agent creation - additional settings."""
        st.markdown("""
        <div class="dashboard-card">
            <h3 style="margin-top:0;">Step 4: Additional Settings</h3>
            <p>Configure advanced settings for your agent to refine its behavior.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Knowledge domains
        st.subheader("Knowledge Domains")
        
        # Get domains from template or initialize empty list
        if not st.session_state.agent_details.get("knowledge_domains"):
            st.session_state.agent_details["knowledge_domains"] = []
        
        # Multi-select for knowledge domains
        available_domains = [
            "General Knowledge", "Academic Research", "Data Analysis", "Programming",
            "Content Creation", "Customer Support", "Finance", "Healthcare", 
            "Legal", "Marketing", "Science", "Technology"
        ]
        
        selected_domains = st.multiselect(
            "Select knowledge domains relevant to this agent",
            options=available_domains,
            default=st.session_state.agent_details.get("knowledge_domains", [])
        )
        st.session_state.agent_details["knowledge_domains"] = selected_domains
        
        # Custom domain input
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_domain = st.text_input("Add custom domain", key="custom_domain")
        with col2:
            if st.button("Add Domain") and custom_domain:
                if custom_domain not in st.session_state.agent_details["knowledge_domains"]:
                    st.session_state.agent_details["knowledge_domains"].append(custom_domain)
                    st.rerun()
        
        # Agent persona
        st.subheader("Agent Persona")
        
        persona_options = ["Professional", "Friendly", "Academic", "Concise", "Detailed", "Casual"]
        persona = st.selectbox(
            "Select the communication style for your agent",
            options=persona_options,
            index=persona_options.index(st.session_state.agent_details.get("persona", "Professional"))
        )
        st.session_state.agent_details["persona"] = persona
        
        # Display different descriptions based on selected persona
        persona_descriptions = {
            "Professional": "Formal, business-oriented communication style with precise language.",
            "Friendly": "Warm, approachable tone that feels more conversational.",
            "Academic": "Scholarly tone with technical language and structured explanations.",
            "Concise": "Brief, to-the-point responses without unnecessary elaboration.",
            "Detailed": "Comprehensive responses with thorough explanations and examples.",
            "Casual": "Relaxed, informal tone with conversational language."
        }
        
        st.info(persona_descriptions.get(persona, ""))
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            # Response length preference
            st.slider(
                "Preferred Response Length",
                min_value=1,
                max_value=5,
                value=st.session_state.agent_details.get("response_length", 3),
                help="1 = Very brief, 5 = Very detailed",
                key="response_length"
            )
            st.session_state.agent_details["response_length"] = st.session_state.response_length
            
            # Temperature setting
            st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.agent_details.get("temperature", 0.7),
                step=0.1,
                help="Lower values = more deterministic, Higher values = more creative",
                key="temperature"
            )
            st.session_state.agent_details["temperature"] = st.session_state.temperature
            
            # Memory settings
            st.toggle(
                "Enable Conversation Memory",
                value=st.session_state.agent_details.get("memory_enabled", True),
                help="Allow agent to remember previous interactions in a conversation",
                key="memory_enabled"
            )
            st.session_state.agent_details["memory_enabled"] = st.session_state.memory_enabled
        
        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.build_step = 3
                st.rerun()
        with col2:
            if st.button("Continue to Review →", type="primary", use_container_width=True):
                st.session_state.build_step = 5
                st.rerun()

    def _render_review_step(self):
        """Render the fifth step of agent creation - review and create."""
        st.markdown("""
        <div class="dashboard-card">
            <h3 style="margin-top:0;">Step 5: Review and Create</h3>
            <p>Review your agent configuration and create your new AI agent.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent details summary
        agent_details = st.session_state.agent_details
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("""
            <div style="background-color: #4361ee; color: white; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                🤖
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader(agent_details.get("name") or "Unnamed Agent")
            st.markdown(f"<p style='color: #666;'>{agent_details.get('description', 'No description provided')}</p>", unsafe_allow_html=True)
        
        st.divider()
        
        # Display summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">Capabilities</h4>
            """, unsafe_allow_html=True)
            
            capabilities = [cap for cap in agent_details.get("capabilities", []) if cap.get("enabled", True)]
            if capabilities:
                for cap in capabilities:
                    st.markdown(f"""
                    <div class="capability-item">
                        <strong>{cap['name']}</strong>
                        <p style="margin: 5px 0 0 0; font-size: 0.9rem;">{cap.get('description', 'No description')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No capabilities selected")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">AI Model</h4>
            """, unsafe_allow_html=True)
            
            provider = agent_details.get("model_provider", "Free Models")
            model = agent_details.get("model", "Gemma-7b")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background-color: #eef2ff; color: #4361ee; 
                    display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: bold;">
                    {provider[0]}
                </div>
                <div>
                    <div style="font-weight: bold;">{provider}</div>
                    <div style="font-size: 0.9rem; color: #666;">{model}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">Knowledge Domains</h4>
            """, unsafe_allow_html=True)
            
            domains = agent_details.get("knowledge_domains", [])
            if domains:
                for domain in domains:
                    st.markdown(f"""
                    <span style="display: inline-block; background-color: #eef2ff; color: #4361ee; padding: 5px 10px; 
                        border-radius: 15px; margin: 0 5px 8px 0; font-size: 0.9rem;">
                        {domain}
                    </span>
                    """, unsafe_allow_html=True)
            else:
                st.info("No knowledge domains specified")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="dashboard-card">
                <h4 style="margin-top:0;">Personality & Settings</h4>
            """, unsafe_allow_html=True)
            
            persona = agent_details.get("persona", "Professional")
            response_length = agent_details.get("response_length", 3)
            temperature = agent_details.get("temperature", 0.7)
            memory_enabled = agent_details.get("memory_enabled", True)
            
            st.markdown(f"""
            <div class="setting-row" style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #666;">Persona:</span>
                <strong>{persona}</strong>
            </div>
            <div class="setting-row" style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #666;">Response Length:</span>
                <strong>{response_length}/5</strong>
            </div>
            <div class="setting-row" style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #666;">Creativity Level:</span>
                <strong>{temperature}</strong>
            </div>
            <div class="setting-row" style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #666;">Conversation Memory:</span>
                <strong>{"Enabled" if memory_enabled else "Disabled"}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Final buttons
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("← Previous", use_container_width=True):
                st.session_state.build_step = 4
                st.rerun()
        
        with col2:
            if st.button("Edit Details", use_container_width=True):
                st.session_state.build_step = 1
                st.rerun()
        
        with col3:
            if st.button("Create Agent", type="primary", use_container_width=True):
                # Create the agent
                with st.spinner("Creating your agent..."):
                    # In a real implementation, pass all the collected details to create_agent
                    # Here we're just using description and name from the wizard
                    if hasattr(st.session_state, 'platform'):
                        result = st.session_state.platform.create_agent(
                            description=agent_details["description"],
                            name=agent_details["name"] if agent_details["name"] else None
                        )
                    else:
                        result = {"success": False, "errors": ["Platform not initialized"]}
                    
                    if result.get("success", False):
                        st.session_state.current_agent = st.session_state.platform.get_agent(result["agent_id"])
                        st.success("Agent created successfully!")
                        
                        # Reset wizard state
                        st.session_state.build_step = 1
                        
                        # Redirect to agent details
                        st.session_state.current_page = "My Agents"
                        st.rerun()
                    else:
                        st.error(f"Error creating agent: {', '.join(result.get('errors', ['Unknown error']))}")

    def _render_agents_page(self):
        """Render the agents page."""
        st.title("🤖 My Agents")
        
        # Get list of agents
        agents = st.session_state.platform.list_agents() if hasattr(st.session_state, 'platform') else []
        
        if not agents:
            st.info("No agents created yet. Go to 'Build Agent' to create one!")
            return
        
        # Allow selecting an agent
        agent_options = {f"{agent['name']} ({agent['id']})": agent['id'] for agent in agents}
        
        selected_agent_id = st.selectbox(
            "Select Agent",
            options=list(agent_options.keys()),
            index=0 if not st.session_state.current_agent else 
                next((i for i, a in enumerate(agents) if a['id'] == st.session_state.current_agent['id']), 0)
        )
        
        agent_id = agent_options[selected_agent_id]
        agent_data = st.session_state.platform.get_agent(agent_id) if hasattr(st.session_state, 'platform') else None
        
        if agent_data:
            # Store selected agent in session state
            st.session_state.current_agent = agent_data
            
            # Show agent details
            st.subheader(agent_data["name"])
            st.write(f"**Purpose:** {agent_data['specification']['purpose']}")
            st.write(f"**Description:** {agent_data['description']}")
            
            # Show capabilities
            st.subheader("Capabilities")
            capabilities = agent_data["specification"]["capabilities"]
            
            # Check if visualization component is available
            if 'render_capability_network' in globals():
                if len(capabilities) > 1:  # Only show network if there are multiple capabilities
                    with st.expander("Capability Network", expanded=True):
                        render_capability_network(capabilities)
            
            for capability in capabilities:
                with st.expander(capability["name"]):
                    st.write(f"**Description:** {capability['description']}")
                    st.write(f"**Dependencies:** {', '.join(capability['dependencies'])}")
                    st.write("**Parameters:**")
                    for param_name, param_value in capability["parameters"].items():
                        st.write(f"- {param_name}: {param_value}")
                    
                    # Show model performance if available
                    if "performance_by_model" in capability and capability["performance_by_model"]:
                        st.write("**Model Performance:**")
                        for model, score in capability["performance_by_model"].items():
                            st.progress(score, text=f"{model}: {score:.2f}")
            
            # Show knowledge domains
            st.subheader("Knowledge Domains")
            st.write(", ".join(agent_data["specification"]["knowledge_domains"]))
            
            # Show recommended models
            if "recommended_models" in agent_data["specification"] and agent_data["specification"]["recommended_models"]:
                st.subheader("Recommended Models")
                for model in agent_data["specification"]["recommended_models"]:
                    st.write(f"- {model}")
            
            # Add metrics visualization if evaluation data is present
            if "evaluation" in agent_data:
                st.subheader("Agent Evaluation")
                eval_data = agent_data.get("evaluation", {})
                if eval_data and isinstance(eval_data, dict):
                    # Extract metric values
                    metrics = {
                        "Completeness": eval_data.get("completeness", 0),
                        "Coherence": eval_data.get("coherence", 0),
                        "Feasibility": eval_data.get("feasibility", 0),
                        "Model Fit": eval_data.get("model_fit", 0),
                        "Overall": eval_data.get("overall_score", 0)
                    }
                    
                    # Check if visualization component is available
                    if 'render_agent_metrics_chart' in globals():
                        render_agent_metrics_chart(metrics)
            
            # Actions
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            if col1.button("Generate Implementation Plan", key="gen_plan"):
                with st.spinner("Generating implementation plan..."):
                    result = st.session_state.platform.generate_implementation_plan(agent_id) if hasattr(st.session_state, 'platform') else {"success": False, "errors": ["Platform not initialized"]}
                    
                    if result["success"]:
                        st.success("Implementation plan generated!")
                        
                        # Display plan summary
                        st.subheader("Implementation Plan")
                        with st.expander("View Implementation Plan"):
                            st.json(result["plan"])
                    else:
                        st.error(f"Error generating plan: {', '.join(result.get('errors', []))}")
            
            if col2.button("Generate Code", key="gen_code"):
                with st.spinner("Generating code..."):
                    result = st.session_state.platform.generate_code(agent_id) if hasattr(st.session_state, 'platform') else {"success": False, "errors": ["Platform not initialized"]}
                    
                    if result["success"]:
                        st.success("Code generated!")
                        
                        # Display code
                        st.subheader("Generated Code")
                        for filename, code in result["code"].items():
                            with st.expander(f"File: {filename}"):
                                st.code(code)
                    else:
                        st.error(f"Error generating code: {', '.join(result.get('errors', []))}")
            
            if col3.button("Deploy Agent", key="deploy"):
                st.session_state.current_page = "Deploy"
                try:
                    # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                    st.rerun()
                except AttributeError:
                    # Fall back to experimental_rerun for older Streamlit versions
                    try:
                        st.experimental_rerun()
                    except:
                        st.error("Failed to rerun the app. Please refresh the page manually.")

    def _render_deploy_page(self):
        """Render the deployment page."""
        st.title("🚀 Deploy Agent")
        
        # Check if we have a selected agent
        if not st.session_state.current_agent:
            # Let the user select an agent
            agents = st.session_state.platform.list_agents()
            
            if not agents:
                st.info("No agents available for deployment. Go to 'Build Agent' to create one!")
                return
            
            agent_options = {f"{agent['name']} ({agent['id']})": agent['id'] for agent in agents}
            
            selected_agent_id = st.selectbox(
                "Select Agent to Deploy",
                options=list(agent_options.keys())
            )
            
            agent_id = agent_options[selected_agent_id]
            agent_data = st.session_state.platform.get_agent(agent_id)
            
            if agent_data:
                st.session_state.current_agent = agent_data
                try:
                    # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                    st.rerun()
                except AttributeError:
                    # Fall back to experimental_rerun for older Streamlit versions
                    try:
                        st.experimental_rerun()
                    except:
                        st.error("Failed to rerun the app. Please refresh the page manually.")
        else:
            # Show deployment options for selected agent
            agent_data = st.session_state.current_agent
            agent_id = agent_data["id"]
            st.subheader(f"Deploy Agent: {agent_data['name']}")
            st.write(f"**Purpose:** {agent_data['specification']['purpose']}")
            
            # Show existing deployments if any
            deployments = [
                d for d in st.session_state.platform.registry["deployments"].values() 
                if d["agent_id"] == agent_id
            ]
            
            if deployments:
                st.subheader("Existing Deployments")
                for deployment in deployments:
                    with st.expander(f"Deployment: {deployment['id']}"):
                        st.write(f"**Status:** {deployment['status']}")
                        st.write(f"**Created at:** {deployment['config'].get('created_at', 'Unknown')}")
                        st.json(deployment['config'])
            
            # New deployment settings
            st.subheader("Create New Deployment")
            
            # Deployment options
            st.write("**Deployment Configuration**")
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("API Access", options=["Public", "Private", "Custom"])
            with col2:
                st.selectbox("Scaling", options=["Auto", "Manual", "Fixed"])
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Memory Allocation (MB)", min_value=128, max_value=4096, value=512, step=128)
                    st.number_input("Timeout (seconds)", min_value=10, max_value=300, value=60, step=10)
                
                with col2:
                    st.number_input("Min Instances", min_value=0, max_value=10, value=1, step=1)
                    st.number_input("Max Instances", min_value=1, max_value=10, value=3, step=1)
            
            # Environment variables
            with st.expander("Environment Variables"):
                st.write("These will be passed to the deployed agent")
                
                # Default env vars
                default_vars = {
                    "OPENAI_API_KEY": "${OPENAI_API_KEY}",  # Use system variable
                    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"  # Use system variable
                }
                
                # Display editable env vars
                updated_vars = {}
                for i, (key, value) in enumerate(default_vars.items()):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        var_name = st.text_input("Name", value=key, key=f"env_name_{i}")
                    with col2:
                        var_value = st.text_input("Value", value=value, type="password" if "KEY" in key else "default", key=f"env_val_{i}")
                    updated_vars[var_name] = var_value
            
            # Deploy button
            if st.button("Deploy Agent", key="deploy_agent"):
                with st.spinner("Deploying agent..."):
                    result = st.session_state.platform.deploy_agent(agent_id)
                    
                    if result["success"]:
                        st.success(f"Agent deployed successfully! Deployment ID: {result['deployment_id']}")
                        
                        # Show deployment details
                        st.subheader("Deployment Details")
                        st.write(f"**Status:** {result['status']}")
                        st.write(f"**Endpoint:** {result['config']['endpoint']}")
                        
                        # Show model assignments if available
                        if "model_assignments" in result["config"]:
                            st.subheader("Model Assignments")
                            for cap_name, model in result["config"]["model_assignments"].items():
                                st.write(f"- {cap_name}: {model}")
                        
                        # Option to test the agent
                        if st.button("Test Deployed Agent", key="test_deployed_agent"):
                            st.session_state.current_page = "Run Agent"
                            try:
                                # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                                st.rerun()
                            except AttributeError:
                                # Fall back to experimental_rerun for older Streamlit versions
                                try:
                                    st.experimental_rerun()
                                except:
                                    st.error("Failed to rerun the app. Please refresh the page manually.")
                    else:
                        st.error(f"Deployment failed: {', '.join(result['errors'])}")

    def _render_run_page(self):
        """Render the run agent page."""
        st.title("💬 Run Agent")
        
        # Get available deployed agents
        deployments = st.session_state.platform.registry["deployments"] if hasattr(st.session_state, 'platform') else {}
        active_deployments = [d for d in deployments.values() if d["status"] == "active"]
        
        if not active_deployments:
            st.info("No active deployments found. Go to 'Deploy' to deploy an agent!")
            return
        
        # Group deployments by agent
        agents_with_deployments = {}
        for deployment in active_deployments:
            agent_id = deployment["agent_id"]
            if agent_id not in agents_with_deployments:
                agent_data = st.session_state.platform.get_agent(agent_id) if hasattr(st.session_state, 'platform') else None
                if agent_data:
                    agents_with_deployments[agent_id] = {
                        "agent": agent_data,
                        "deployments": []
                    }
            
            if agent_id in agents_with_deployments:
                agents_with_deployments[agent_id]["deployments"].append(deployment)
        
        # Let the user select an agent
        agent_options = {}
        for agent_id, data in agents_with_deployments.items():
            agent_name = data["agent"]["name"]
            agent_options[f"{agent_name} ({agent_id})"] = agent_id
        
        if not agent_options:
            st.error("Failed to load any valid deployments. Check your configuration.")
            return
            
        selected_agent_key = st.selectbox(
            "Select Agent",
            options=list(agent_options.keys())
        )
        
        selected_agent_id = agent_options[selected_agent_key]
        selected_agent_data = agents_with_deployments[selected_agent_id]
        
        # Show agent info
        st.subheader(f"Agent: {selected_agent_data['agent']['name']}")
        st.write(f"**Purpose:** {selected_agent_data['agent']['specification']['purpose']}")
        st.write(f"**Description:** {selected_agent_data['agent']['description']}")
        
        # Chat interface
        st.divider()
        st.subheader("Chat with Agent")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
                
        # Chat input
        user_query = st.chat_input("Type your message here...")
        
        if user_query:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            # Display user message
            st.chat_message("user").write(user_query)
            
            # Query agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if hasattr(st.session_state, 'platform'):
                        result = st.session_state.platform.query_agent(
                            agent_id=selected_agent_id,
                            query=user_query
                        )
                    else:
                        result = {"success": False, "errors": ["Platform not initialized"]}
                    
                    if result["success"]:
                        response = result["response"]
                        st.write(response)
                        
                        # Add assistant message to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Display metadata in expander
                        with st.expander("Response Metadata"):
                            if "metadata" in result:
                                st.write(f"**Model:** {result['metadata'].get('model', 'Unknown')}")
                                st.write(f"**Processing Time:** {result['metadata'].get('processing_time', 0):.2f}s")
                                if "usage" in result["metadata"]:
                                    st.write(f"**Tokens Used:** {result['metadata']['usage'].get('total_tokens', 0)}")
                    else:
                        st.error(f"Error: {', '.join(result.get('errors', []))}")

    def _render_settings_page(self):
        """Render the settings page."""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["API Keys", "Model Settings", "Database", "System"])
        
        # API Keys tab
        with tabs[0]:
            st.subheader("API Keys")
            
            # OpenAI
            openai_key = st.text_input(
                "OpenAI API Key", 
                value=st.session_state.api_keys.get('openai', ''),
                type="password"
            )
            
            # Anthropic
            anthropic_key = st.text_input(
                "Anthropic API Key", 
                value=st.session_state.api_keys.get('anthropic', ''),
                type="password"
            )
            
            # Cohere
            cohere_key = st.text_input(
                "Cohere API Key", 
                value=st.session_state.api_keys.get('cohere', ''),
                type="password"
            )
            
            if st.button("Save API Keys"):
                # Update session state
                st.session_state.api_keys['openai'] = openai_key
                st.session_state.api_keys['anthropic'] = anthropic_key
                st.session_state.api_keys['cohere'] = cohere_key
                
                # Update platform config
                if hasattr(st.session_state, 'platform'):
                    st.session_state.platform.config["api_keys"]["openai"] = openai_key
                    st.session_state.platform.config["api_keys"]["anthropic"] = anthropic_key
                    st.session_state.platform.config["api_keys"]["cohere"] = cohere_key
                
                # Reinitialize platform
                self._initialize_platform()
                
                st.success("API keys saved and platform reinitialized!")
        
        # Model Settings tab
        with tabs[1]:
            st.subheader("Model Settings")
            
            # Default models
            st.write("**Default Models for Capabilities**")
            
            # Get all model providers
            if hasattr(st.session_state, 'platform') and hasattr(st.session_state.platform, "components"):
                agent_builder = st.session_state.platform.components.get("agent_builder", {})
                if "builder" in agent_builder:
                    builder = agent_builder["builder"]
                    
                    # Check if builder has model_registry attribute
                    if hasattr(builder, "model_registry"):
                        # OpenAI models
                        if "openai" in builder.model_registry.providers:
                            try:
                                openai_models = builder.model_registry.get_available_models("openai")
                                if openai_models:
                                    default_openai = st.selectbox(
                                        "Default OpenAI Model", 
                                        options=openai_models,
                                        index=openai_models.index(builder.default_model) if builder.default_model in openai_models else 0
                                    )
                                    
                                    if st.button("Set Default OpenAI Model"):
                                        builder.default_model = default_openai
                                        st.success(f"Default OpenAI model set to {default_openai}")
                            except Exception as e:
                                st.warning(f"Error loading OpenAI models: {e}")
                        
                        # Anthropic models
                        if "anthropic" in builder.model_registry.providers:
                            try:
                                anthropic_models = builder.model_registry.get_available_models("anthropic")
                                if anthropic_models:
                                    default_anthropic = st.selectbox(
                                        "Default Anthropic Model", 
                                        options=anthropic_models
                                    )
                                    
                                    if st.button("Set Default Anthropic Model"):
                                        builder.model_registry.providers["anthropic"].default_model = default_anthropic
                                        st.success(f"Default Anthropic model set to {default_anthropic}")
                            except Exception as e:
                                st.warning(f"Error loading Anthropic models: {e}")
                    else:
                        st.info("Model registry not available. Using default configuration.")
                        
                        # Show basic model configuration
                        st.write("Default model configuration:")
                        if hasattr(st.session_state.platform, "config"):
                            default_models = st.session_state.platform.config.get("default_models", {})
                            for provider, model in default_models.items():
                                st.write(f"- {provider}: {model}")

        # Database tab
        with tabs[2]:
            st.subheader("Database Settings")
            
            if hasattr(st.session_state, 'platform'):
                db_info = st.session_state.platform.components.get("database", {})
                if db_info:
                    st.write(f"**Current Database:** {db_info.get('type', 'Unknown')}")
                    
                    if db_info.get('type') == 'sqlite':
                        st.write(f"**Database Path:** {st.session_state.platform.config['database_config']['sqlite']['db_path']}")
                    elif db_info.get('type') == 'postgres':
                        postgres_config = st.session_state.platform.config['database_config']['postgres']
                        st.write(f"**Host:** {postgres_config['host']}:{postgres_config['port']}")
                        st.write(f"**Database:** {postgres_config['database']}")
                    
                    # Vector database
                    vector_db = db_info.get('vector_db', {})
                    if vector_db:
                        st.write(f"**Vector Database:** {vector_db.get('type', 'Unknown')}")
                        
                        if vector_db.get('type') == 'chromadb':
                            st.write(f"**Path:** {st.session_state.platform.config['vector_database']['path']}")
                            
                            collections = vector_db.get('collections', {})
                            if collections:
                                st.write("**Collections:**")
                                for name, collection in collections.items():
                                    if collection:
                                        st.write(f"- {name}")
                else:
                    st.warning("Database information not available")
        
        # System tab
        with tabs[3]:
            st.subheader("System Information")
            
            if hasattr(st.session_state, 'platform'):
                status = st.session_state.platform.get_status()
                
                st.write(f"**Version:** {status['version']}")
                
                # Component status
                st.write("**Component Status:**")
                for component, is_active in status["components"].items():
                    st.write(f"- {component.capitalize()}: {'✅ Active' if is_active else '❌ Inactive'}")
                
                # Registry counts
                st.write("**Registry Counts:**")
                for name, count in status["registry"].items():
                    st.write(f"- {name.replace('_', ' ').capitalize()}: {count}")
                
                # System metrics if available
                if "metrics" in status:
                    st.write("**System Metrics:**")
                    metrics = status["metrics"]
                    for metric_name, value in metrics.items():
                        st.metric(metric_name.capitalize(), f"{value}%")
                
                # Advanced actions
                with st.expander("Advanced Actions"):
                    if st.button("Reinitialize Platform", key="reinitialize_platform"):
                        self._initialize_platform()
                        st.success("Platform reinitialized!")
                    if st.button("Clear Cache", key="clear_cache"):
                        st.cache_data.clear()
                        st.success("Cache cleared!")
                        
                    if st.button("Reset All Data", key="reset_all_data"):
                        st.warning("This will delete all agents, capabilities, and deployments!")
                        confirm = st.checkbox("I understand that all data will be lost")
                        
                        if confirm and st.button("Confirm Reset", key="confirm_reset"):
                            # In a real app, would need to implement data deletion logic
                            try:
                                data_dir = st.session_state.platform.config["data_directory"]
                                import shutil
                                
                                # Only delete contents, not the directory itself
                                for item in os.listdir(data_dir):
                                    item_path = os.path.join(data_dir, item)
                                    if os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                                    else:
                                        os.remove(item_path)
                                
                                # Reinitialize platform
                                self._initialize_platform()
                                st.success("All data has been reset!")
                                
                                try:
                                    # Use the new st.rerun() if available (Streamlit >= 1.27.0)
                                    st.rerun()
                                except AttributeError:
                                    # Fall back to experimental_rerun for older Streamlit versions
                                    try:
                                        st.experimental_rerun()
                                    except:
                                        st.error("Failed to rerun the app. Please refresh the page manually.")
                            except Exception as e:
                                st.error(f"Error resetting data: {e}")

    def _render_status_page(self):
        """Render the status page with monitoring data."""
        st.title("📊 System Status")
        
        if hasattr(st.session_state, 'platform'):
            # Get status information
            status = st.session_state.platform.get_status()
            
            # Display uptime and version
            st.subheader("System Info")
            col1, col2 = st.columns(2)
            col1.metric("Version", status['version'])
            col2.metric("Uptime", f"{status.get('uptime_seconds', 0) // 3600}h {(status.get('uptime_seconds', 0) % 3600) // 60}m")
            
            # Component status
            st.subheader("Components")
            
            cols = st.columns(len(status["components"]))
            for i, (component, is_active) in enumerate(status["components"].items()):
                cols[i].metric(
                    component.capitalize(),
                    "Active" if is_active else "Inactive",
                    delta="✓" if is_active else "✗",
                    delta_color="normal" if is_active else "inverse"
                )
            
            # System metrics if available
            if "metrics" in status and status["metrics"]:
                st.subheader("System Metrics")
                
                cols = st.columns(len(status["metrics"]))
                for i, (metric, value) in enumerate(status["metrics"].items()):
                    cols[i].metric(metric.capitalize(), f"{value:.1f}%")
                    
                    # Add progress bars
                    if metric == "cpu":
                        cols[i].progress(value / 100)
                    elif metric == "memory":
                        cols[i].progress(value / 100)
                    elif metric == "disk":
                        cols[i].progress(value / 100)
            
            # Registry stats
            st.subheader("Content")
            
            cols = st.columns(len(status["registry"]))
            for i, (item_type, count) in enumerate(status["registry"].items()):
                cols[i].metric(item_type.replace("_", " ").capitalize(), count)
            
            # Error log if available
            error_log_path = os.path.join(
                st.session_state.platform.config["data_directory"],
                "logs",
                "errors.log"
            )
            
            if os.path.exists(error_log_path) and os.path.getsize(error_log_path) > 0:
                st.subheader("Recent Errors")
                try:
                    with open(error_log_path, "r") as f:
                        # Read last 10 lines
                        lines = f.readlines()[-10:]
                        errors = []
                        
                        for line in lines:
                            try:
                                error_data = json.loads(line.strip())
                                errors.append(error_data)
                            except:
                                # If not JSON, just show the raw line
                                st.text(line.strip())
                    
                    # Display parsed errors
                    for error in errors:
                        with st.expander(f"{error.get('component', 'Unknown')}: {error.get('error_type', 'Error')}"):
                            st.write(f"**Message:** {error.get('error_msg', 'Unknown error')}")
                            st.write(f"**Time:** {error.get('timestamp', 'Unknown')}")
                            if "context" in error and error["context"]:
                                st.json(error["context"])
                except Exception as e:
                    st.error(f"Error loading error log: {e}")
        else:
            st.error("Platform not initialized properly.")


def run_app():
    """Run the integrated UI application."""
    # No need to call st.set_page_config() here anymore
    
    # Show error if imports failed
    if UnifiedPlatform is None:
        st.error("Failed to import required modules. Make sure all dependencies are installed.")
    
    # Get theme from session state or default to dark
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
        
    # Apply theme if function exists
    if apply_theme is not None:
        apply_theme(st.session_state.theme)
    
    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create and render the UI
    ui = IntegratedUI()
    ui.render()

if __name__ == "__main__":
    run_app()