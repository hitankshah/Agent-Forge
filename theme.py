"""
Theme handling for the Agent Platform UI.
"""

import streamlit as st
import json
import os
from typing import Dict, Any, Optional

# Define available themes
THEMES = {
    "dark": {
        "name": "Dark",
        "primaryColor": "#4b6fff",
        "backgroundColor": "#0e1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#fafafa",
        "font": "sans-serif",
        "css": """
        /* Dark theme extra styles */
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #10141f 0%, #1c2033 100%);
        }
        
        .dashboard-card {
            background-color: #1c2033 !important;
            border-color: #4b6fff !important;
        }
        
        .agent-card {
            background: linear-gradient(135deg, #1c2033 0%, #10141f 100%) !important;
            border-left-color: #4b6fff !important;
        }
        
        .st-bq {
            background-color: #262730 !important;
            border-left-color: #4b6fff !important;
        }
        
        .stButton > button {
            background-color: #4b6fff !important;
            color: white !important;
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #4b6fff 0%, #6e8dfb 100%) !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0 !important;
        }
        
        code {
            background-color: #262730 !important;
            border-color: #444 !important;
        }
        """
    },
    "light": {
        "name": "Light",
        "primaryColor": "#4b6fff",
        "backgroundColor": "#f5f7fa",
        "secondaryBackgroundColor": "#ffffff",
        "textColor": "#31333f",
        "font": "sans-serif",
        "css": """
        /* Light theme extra styles */
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #f5f7fa 0%, #e8eaed 100%);
        }
        
        .dashboard-card {
            background-color: white !important;
            border-color: #4b6fff !important;
        }
        
        .stButton > button:hover {
            border-color: #4b6fff !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #1a2942 !important;
        }
        """
    },
    "cyberpunk": {
        "name": "Cyberpunk",
        "primaryColor": "#ff0055",
        "backgroundColor": "#031d44",
        "secondaryBackgroundColor": "#04395e",
        "textColor": "#f0cb35",
        "font": "monospace",
        "css": """
        /* Cyberpunk theme extra styles */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        
        * {
            font-family: 'Share Tech Mono', monospace !important;
        }
        
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #031d44 0%, #04395e 100%);
        }
        
        .dashboard-card {
            background-color: #04395e !important;
            border-color: #ff0055 !important;
            border-left: 0px !important;
            border-top: 4px solid #ff0055 !important;
            box-shadow: 0 0 10px #ff005580 !important;
        }
        
        .agent-card {
            background: #04395e !important;
            border-left: 5px solid #ff0055 !important;
            box-shadow: 0 0 10px #ff005580 !important;
        }
        
        .stButton > button {
            background-color: #ff0055 !important;
            color: #f0cb35 !important;
            border: 2px solid #f0cb35 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #f0cb35 !important;
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #ff0055 0%, #ff5e8a 100%) !important;
        }
        
        /* Terminal-like text */
        p, span, div {
            color: #f0cb35 !important;
        }
        
        /* Code blocks */
        code {
            background-color: #000 !important;
            color: #0f0 !important;
            border-color: #f0cb35 !important;
        }
        
        /* Glitch animation for headings */
        @keyframes glitch {
            0% { transform: translate(0); }
            20% { transform: translate(-2px, 2px); }
            40% { transform: translate(-2px, -2px); }
            60% { transform: translate(2px, 2px); }
            80% { transform: translate(2px, -2px); }
            100% { transform: translate(0); }
        }
        
        h1:hover, h2:hover {
            animation: glitch 0.3s linear infinite;
            text-shadow: 0 0 10px #ff0055;
        }
        """
    }
}

def apply_theme(theme_name: str = "dark") -> None:
    """
    Apply a theme to the Streamlit app.
    
    Args:
        theme_name: The name of the theme to apply
    """
    # Ensure the theme exists, default to dark if not
    if theme_name not in THEMES:
        theme_name = "dark"
    
    # Get theme data
    theme = THEMES[theme_name]
    
    # Apply basic theme settings
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {theme["primaryColor"]};
            --background-color: {theme["backgroundColor"]};
            --secondary-background-color: {theme["secondaryBackgroundColor"]};
            --text-color: {theme["textColor"]};
            --font: {theme["font"]};
        }}
        
        .stApp {{
            background-color: {theme["backgroundColor"]};
            color: {theme["textColor"]};
            font-family: {theme["font"]};
        }}
        
        /* Apply theme's custom CSS */
        {theme["css"]}
    </style>
    """, unsafe_allow_html=True)

def get_available_themes() -> list:
    """
    Get a list of available theme names.
    
    Returns:
        List of theme names
    """
    return list(THEMES.keys())
