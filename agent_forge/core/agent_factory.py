from typing import Dict, Any, List, Optional, Type
from loguru import logger
from agent_forge.core.agent import Agent
from agent_forge.integrations.llm import LLMProvider
from agent_forge.utils.templates import load_template

class AgentFactory:
    """Factory class to create and customize agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the agent factory with configuration."""
        self.config = config
        self.llm_provider = LLMProvider(config.get("llm", {}))
        self.registered_agent_types: Dict[str, Type[Agent]] = {}
    
    def register_agent_type(self, name: str, agent_class: Type[Agent]) -> None:
        """Register a new agent type."""
        self.registered_agent_types[name] = agent_class
        logger.info(f"Registered agent type: {name}")
    
    def create_agent(self, 
                    name: str, 
                    description: str, 
                    agent_type: str = "default",
                    capabilities: Optional[List[str]] = None, 
                    config: Optional[Dict[str, Any]] = None) -> Agent:
        """Create a new agent with the specified parameters."""
        logger.info(f"Creating new agent: {name} of type {agent_type}")
        
        # Get the agent class based on type
        agent_class = self.registered_agent_types.get(agent_type)
        if not agent_class:
            logger.warning(f"Agent type '{agent_type}' not found, using default Agent class")
            agent_class = Agent
        
        # Create the agent instance
        agent = agent_class(
            name=name,
            description=description,
            capabilities=capabilities or [],
            config=config or {}
        )
        
        return agent
    
    def generate_agent_from_description(self, description: str) -> Agent:
        """Generate an agent automatically from a text description."""
        logger.info(f"Generating agent from description: {description[:50]}...")
        
        # Load the agent generation template
        template = load_template("agent_generation")
        
        # Use LLM to generate agent specification
        agent_spec = self.llm_provider.generate(
            template=template,
            inputs={"description": description}
        )
        
        # Create the agent based on the generated specification
        try:
            # Parse the LLM response into structured data
            parsed_spec = self._parse_agent_spec(agent_spec)
            
            return self.create_agent(
                name=parsed_spec.get("name", "Generated Agent"),
                description=parsed_spec.get("description", description),
                capabilities=parsed_spec.get("capabilities", []),
                config=parsed_spec.get("config", {})
            )
        except Exception as e:
            logger.error(f"Failed to create agent from description: {str(e)}")
            # Fallback to simple agent
            return self.create_agent(
                name="Generated Agent",
                description=description
            )
    
    def _parse_agent_spec(self, spec_text: str) -> Dict[str, Any]:
        """Parse the LLM-generated agent specification into structured data."""
        # In a real implementation, this would have more robust parsing
        # This is a simplified version
        try:
            # Try to parse as JSON if possible
            import json
            return json.loads(spec_text)
        except:
            # Fallback to simple parsing
            lines = spec_text.split("\n")
            result = {"capabilities": [], "config": {}}
            
            for line in lines:
                if line.startswith("Name:"):
                    result["name"] = line.replace("Name:", "").strip()
                elif line.startswith("Description:"):
                    result["description"] = line.replace("Description:", "").strip()
                elif line.startswith("- "):
                    result["capabilities"].append(line.replace("- ", "").strip())
            
            return result
