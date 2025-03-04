import os
from typing import Dict, Any
from loguru import logger

# Dictionary of built-in templates
DEFAULT_TEMPLATES = {
    "agent_generation": """
    You are an AI agent designer. Your task is to create a new AI agent based on the following description:
    
    Description: {description}
    
    Please design an agent with the following details:
    1. A clear name for the agent
    2. A concise description of what the agent does
    3. A list of capabilities the agent should have
    4. Any specific configuration parameters
    
    Format your response as a JSON object with the following structure:
    {
        "name": "Agent name",
        "description": "Agent description",
        "capabilities": ["capability1", "capability2", ...],
        "config": {
            "param1": value1,
            "param2": value2
        }
    }
    """,
    
    "agent_execution": """
    You are an AI agent named {agent_name} with the following description:
    {agent_description}
    
    You have these capabilities:
    {agent_capabilities}
    
    Task: {task}
    
    Please execute this task to the best of your abilities, taking into account your specific capabilities.
    """,
    
    "agent_reflection": """
    You are an AI agent named {agent_name}.
    
    Review your recent actions and outcomes:
    {recent_history}
    
    Reflect on your performance by analyzing:
    1. What went well?
    2. What could be improved?
    3. What new strategies could you implement?
    4. How can you better achieve your objectives?
    
    Format your self-reflection as a structured analysis.
    """
}

def load_template(template_name: str) -> str:
    """
    Load a template by name. First checks if a custom template file exists,
    otherwise falls back to built-in templates.
    
    Args:
        template_name: Name of the template to load
        
    Returns:
        String containing the template text
    """
    # Check for custom template file
    template_path = os.path.join("templates", f"{template_name}.txt")
    if os.path.exists(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading template file {template_path}: {str(e)}")
    
    # Fall back to built-in templates
    if template_name in DEFAULT_TEMPLATES:
        return DEFAULT_TEMPLATES[template_name]
    
    # Template not found
    logger.warning(f"Template '{template_name}' not found, returning empty string")
    return ""
