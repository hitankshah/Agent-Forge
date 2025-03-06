"""
Unified server launcher that provides options to start either the legacy UI
or the new integrated UI with advanced capabilities.
"""

import os
import sys
import argparse
import logging
import threading
import time
import subprocess
from pathlib import Path

# Import logging setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.setup_logging import setup_logging

# Configure logging
logger = setup_logging()
logger = logging.getLogger('UnifiedServer')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Start Unified Agent Platform')
    parser.add_argument('--ui', choices=['legacy', 'integrated'], default='integrated',
                      help='UI version to launch (legacy or integrated)')
    parser.add_argument('--settings', action='store_true', help='Start directly in settings view')
    parser.add_argument('--port', type=int, default=8501, help='Port for the Streamlit UI')
    parser.add_argument('--api', action='store_true', default=False, help='Launch API server')
    parser.add_argument('--api-port', type=int, default=8000, help='Port for the API server')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def start_streamlit(script_path, port=8501, settings=False):
    """Start a Streamlit server using subprocess instead of direct imports"""
    try:
        # Build the command
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)]
        
        # Add settings flag if requested
        if settings:
            cmd.append("--")
            cmd.append("--settings")
            
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        logger.info(f"Starting Streamlit server on port {port} with script {script_path}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start threads to monitor output
        def log_stream(stream, log_level):
            for line in stream:
                line = line.strip()
                # More robust detection of INFO level messages
                if any(pattern in line for pattern in ["INFO:", ":INFO:", "- INFO -", "INFO -", "- INFO"]):
                    logger.info(f"Streamlit: {line}")
                    continue
                    
                # Error detection
                if log_level == logging.ERROR:
                    if any(pattern in line for pattern in ["Error", "ERROR", "Exception", "Traceback", "Failed"]):
                        logger.error(f"Streamlit error: {line}")
                    else:
                        # If not specifically an error message, log as info
                        logger.info(f"Streamlit: {line}")
                else:
                    logger.info(f"Streamlit: {line}")
            
        threading.Thread(target=log_stream, args=(process.stdout, logging.INFO), daemon=True).start()
        threading.Thread(target=log_stream, args=(process.stderr, logging.ERROR), daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error starting Streamlit subprocess: {e}")
        return None

def start_api_server(port=8000):
    """Start the FastAPI server"""
    try:
        cmd = [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", str(port)]
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        logger.info(f"Starting API server on port {port}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start threads to monitor output
        def log_stream(stream, log_level):
            for line in stream:
                if log_level == logging.INFO:
                    logger.info(f"API Server: {line.strip()}")
                else:
                    logger.error(f"API Server error: {line.strip()}")
            
        threading.Thread(target=log_stream, args=(process.stdout, logging.INFO), daemon=True).start()
        threading.Thread(target=log_stream, args=(process.stderr, logging.ERROR), daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return None

def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        "data",
        "data/agents",
        "data/deployments",
        "data/implementations",
        "data/logs",
        "config"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def ensure_config_file():
    """Ensure configuration file exists"""
    config_dir = "config"
    config_path = os.path.join(config_dir, "integration_config.json")
    
    if not os.path.exists(config_path):
        import json
        
        default_config = {
            "data_directory": "data",
            "default_database": "sqlite",
            "database_config": {
                "sqlite": {
                    "db_path": "data/unified_platform.db"
                }
            },
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "cohere": os.environ.get("COHERE_API_KEY", "")
            },
            "default_models": {
                "openai": "gpt-4",
                "anthropic": "claude-2"
            },
            "vector_database": {
                "type": "chromadb",
                "path": "data/vector_db"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.info(f"Created default configuration at {config_path}")

def main():
    """Main entry point for the server"""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure required directories and files exist
    ensure_directories()
    ensure_config_file()
    
    # Start the appropriate UI as a subprocess
    streamlit_process = None
    api_process = None
    
    try:
        # Determine which UI script to run
        if args.ui == 'integrated':
            script_path = os.path.join(os.path.dirname(__file__), "integrated_ui.py")
            if not os.path.exists(script_path):
                logger.error(f"Integrated UI script not found at {script_path}")
                logger.info("Falling back to legacy UI")
                script_path = os.path.join(os.path.dirname(__file__), "start_server.py")
        else:
            script_path = os.path.join(os.path.dirname(__file__), "start_server.py")
        
        streamlit_process = start_streamlit(script_path, args.port, args.settings)
        
        if streamlit_process:
            logger.info(f"Started {args.ui} UI at http://localhost:{args.port}")
        else:
            logger.error(f"Failed to start {args.ui} UI")
            return
        
        # Start API server if requested
        if args.api:
            api_script_path = os.path.join(os.path.dirname(__file__), "api_server.py")
            
            # Check if API server script exists
            if not os.path.exists(api_script_path):
                logger.error(f"API server script not found at {api_script_path}")
                logger.error("Creating minimal API server...")
                
                # Create minimal API server script
                with open(api_script_path, 'w') as f:
                    f.write("""
# filepath: /api_server.py
from fastapi import FastAPI
from unified_platform import UnifiedPlatform

app = FastAPI(title="Agent Platform API", description="API for Agent Platform", version="1.0.0")
platform = UnifiedPlatform()

@app.get("/")
def read_root():
    return {"message": "Agent Platform API", "version": "1.0.0"}

@app.get("/agents")
def list_agents():
    return {"agents": platform.list_agents()}
""")
                logger.info(f"Created minimal API server at {api_script_path}")
            
            # Start API server
            api_process = start_api_server(args.api_port)
            
            if api_process:
                logger.info(f"Started API server at http://localhost:{args.api_port}")
            else:
                logger.error("Failed to start API server")
        
        # Wait for processes to finish or user interrupt
        try:
            while True:
                # Check if streamlit is still running
                if streamlit_process and streamlit_process.poll() is not None:
                    logger.error("Streamlit process has terminated")
                    break
                
                # Check if API server is still running (if started)
                if api_process and api_process.poll() is not None:
                    logger.error("API server process has terminated")
                    break
                    
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Terminate processes on exit
        if streamlit_process and streamlit_process.poll() is None:
            try:
                streamlit_process.terminate()
                logger.info("Streamlit process terminated")
            except Exception as e:
                logger.error(f"Error terminating Streamlit process: {e}")
                
        if api_process and api_process.poll() is None:
            try:
                api_process.terminate()
                logger.info("API server process terminated")
            except Exception as e:
                logger.error(f"Error terminating API server process: {e}")
    
    logger.info("Server shutting down")

if __name__ == "__main__":
    main()