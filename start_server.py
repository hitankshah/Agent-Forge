import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Streamlit runner functions
from agent_forge.ui.app import run_ui
from agent_forge.ui.settings_ui import SettingsUI, render_settings_page

def main():
    """Main entry point for starting the server."""
    # Check if we're running the settings UI or main app
    if len(sys.argv) > 1 and sys.argv[1] == "--settings":
        print("Starting Settings UI...")
        render_settings_page()
        
        # Initialize and render all sections of settings
        settings_ui = SettingsUI()
        settings_ui.render_env_manager()
        settings_ui.render_model_configuration()
        settings_ui.render_advanced_settings()
        settings_ui.render_embedding_demo()
    else:
        print("Starting Agent Forge UI...")
        # Run the main application UI
        run_ui()

if __name__ == "__main__":
    main()
