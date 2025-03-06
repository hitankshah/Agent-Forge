"""
API Server for Agent Platform
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import sys
import json
import time

# Add the parent directory to the path for imports
parent_dir = str(Path(__file__).parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Create FastAPI app
app = FastAPI(
    title="Agent Platform API",
    description="API for managing AI agents with Agent Platform",
    version="1.0.0"
)

# Import platform after path setup
try:
    from unified_platform import UnifiedPlatform
    platform = UnifiedPlatform()
except ImportError:
    print("Failed to import UnifiedPlatform. Initializing with basic API functionality.")
    platform = None

@app.get("/")
def read_root():
    """Root endpoint with API info."""
    return {
        "message": "Agent Platform API",
        "version": "1.0.0",
        "status": "operational",
        "platform_loaded": platform is not None
    }

@app.get("/agents")
def list_agents():
    """List all available agents."""
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    try:
        agents = platform.list_agents()
        return {"agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.get("/agents/{agent_id}")
def get_agent(agent_id: str):
    """Get details for a specific agent."""
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    try:
        agent = platform.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        return {"agent": agent}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {str(e)}")

@app.post("/agents/{agent_id}/deploy")
def deploy_agent(agent_id: str):
    """Deploy an agent."""
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    try:
        result = platform.deploy_agent(agent_id)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=", ".join(result["errors"]))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deploying agent: {str(e)}")

@app.post("/agents/{agent_id}/query")
def query_agent(agent_id: str, request: Dict[str, Any]):
    """Query an agent with a message."""
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        parameters = request.get("parameters", {})
        
        result = platform.query_agent(
            agent_id=agent_id,
            query=query,
            parameters=parameters
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=", ".join(result["errors"]))
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying agent: {str(e)}")
