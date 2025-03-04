import os
import argparse
from loguru import logger

from agent_forge.core.agent_factory import AgentFactory
from agent_forge.utils.config import load_config
from agent_forge.ui.app import run_ui

def main():
    """Main entry point for Agent Forge."""
    parser = argparse.ArgumentParser(description="Agent Forge - AI Agent Builder")
    parser.add_argument("--ui", action="store_true", help="Launch the UI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.ui:
        # Start the Streamlit UI
        run_ui()
    else:
        # CLI mode
        agent_factory = AgentFactory(config)
        logger.info("Agent Forge initialized in CLI mode. Use --ui flag to launch the UI.")

if __name__ == "__main__":
    main()
