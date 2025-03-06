"""
Connector module that bridges the Agent Builder library with other components.
This provides a clean API for applications to use the Agent Builder functionality.
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from agent_builder import AgentBuilder, AgentSpecification, AgentCapability
from agent_builder.model_providers import OpenAIProvider, AnthropicProvider

class AgentForgeConnector:
    """Connector class for integrating Agent Builder with applications"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the connector with optional configuration.
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize the builder with API keys from config or environment
        self.builder = AgentBuilder(
            openai_api_key=self.config.get('openai_api_key', os.environ.get('OPENAI_API_KEY')),
            anthropic_api_key=self.config.get('anthropic_api_key', os.environ.get('ANTHROPIC_API_KEY')),
            default_model=self.config.get('default_model', 'gpt-4')
        )
        
        # Set up storage paths
        self.agents_dir = Path(self.config.get('agents_dir', 'agents'))
        self.agents_dir.mkdir(exist_ok=True)
    
    def create_agent_from_description(self, description: str) -> Dict[str, Any]:
        """
        Create an agent from a natural language description.
        
        Args:
            description: Natural language description of the agent
            
        Returns:
            Dictionary with agent data including the specification
        """
        # Analyze requirements
        requirements = self.builder.analyze_requirements(description)
        
        if "error" in requirements:
            return {"success": False, "error": requirements["error"]}
        
        # Create specification
        spec = self.builder.create_agent_specification(requirements)
        
        # Evaluate the design
        evaluation = self.builder.evaluate_agent_design(spec)
        
        # Return complete results
        return {
            "success": True,
            "agent": spec.to_dict(),
            "evaluation": evaluation,
            "requirements": requirements
        }
    
    def create_implementation_plan(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate implementation plan for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with implementation plan
        """
        spec = self.load_agent(agent_id)
        if not spec:
            return {"success": False, "error": f"Agent with ID '{agent_id}' not found"}
        
        plan = self.builder.generate_implementation_plan(spec)
        return {
            "success": True,
            "agent_id": agent_id,
            "implementation_plan": plan
        }
    
    def generate_code(self, agent_id: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate code for an agent based on implementation plan.
        
        Args:
            agent_id: ID of the agent
            plan: Optional implementation plan (if not provided, will be generated)
            
        Returns:
            Dictionary with code skeletons
        """
        spec = self.load_agent(agent_id)
        if not spec:
            return {"success": False, "error": f"Agent with ID '{agent_id}' not found"}
        
        if not plan:
            plan = self.builder.generate_implementation_plan(spec)
        
        code = self.builder.generate_code_skeleton(plan)
        return {
            "success": True,
            "agent_id": agent_id,
            "code": code
        }
    
    def save_agent(self, agent_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save an agent specification.
        
        Args:
            agent_spec: Agent specification as dictionary
            
        Returns:
            Dictionary with save status
        """
        try:
            spec = AgentSpecification.from_dict(agent_spec)
            agent_id = spec.name.lower().replace(" ", "_")
            filepath = self.agents_dir / f"{agent_id}.json"
            
            self.builder.save_agent_specification(spec, str(filepath))
            return {
                "success": True,
                "agent_id": agent_id,
                "filepath": str(filepath)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def load_agent(self, agent_id: str) -> Optional[AgentSpecification]:
        """
        Load an agent specification by ID.
        
        Args:
            agent_id: ID of the agent to load
            
        Returns:
            AgentSpecification if found, None otherwise
        """
        filepath = self.agents_dir / f"{agent_id}.json"
        if filepath.exists():
            return self.builder.load_agent_specification(str(filepath))
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents.
        
        Returns:
            List of agent summaries
        """
        agents = []
        
        if self.agents_dir.exists():
            for filepath in self.agents_dir.glob("*.json"):
                agent_id = filepath.stem
                spec = self.builder.load_agent_specification(str(filepath))
                if spec:
                    agents.append({
                        "id": agent_id,
                        "name": spec.name,
                        "description": spec.description,
                        "purpose": spec.purpose,
                        "capabilities_count": len(spec.capabilities),
                        "recommended_models": spec.recommended_models
                    })
        
        return agents
    
    def optimize_agent_models(self, agent_id: str, budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize model selection for an agent.
        
        Args:
            agent_id: ID of the agent
            budget: Optional budget constraint
            
        Returns:
            Dictionary with model assignments
        """
        spec = self.load_agent(agent_id)
        if not spec:
            return {"success": False, "error": f"Agent with ID '{agent_id}' not found"}
        
        model_assignments = self.builder.optimize_model_selection(spec, budget)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "model_assignments": model_assignments
        }
