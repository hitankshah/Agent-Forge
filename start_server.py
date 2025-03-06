import streamlit as st
import argparse
import os
import sys
from pathlib import Path

# Set environment variables early to avoid conflicts
os.environ["PYTHONUNBUFFERED"] = "1"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Start Agent Forge server')
    parser.add_argument('--settings', action='store_true', help='Start directly in settings view')
    return parser.parse_args()

def main():
    """Main function to run the application"""
    # Set page config
    st.set_page_config(
        page_title="Agent Forge",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    
    # Import here to avoid circular imports and asyncio conflicts
    from agent_builder.ui import AgentBuilderUI
    
    # Create and run the UI
    ui = AgentBuilderUI()
    
    # Handle direct navigation to settings if requested
    args = parse_args()
    if args.settings and 'page_selection' not in st.session_state:
        st.session_state.page_selection = "Settings"
    
    ui.run()

if __name__ == "__main__":
    main()
