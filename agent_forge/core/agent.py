from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4, UUID
import time
from loguru import logger

class AgentMemory(BaseModel):
    """Agent memory storage."""
    short_term: List[Dict[str, Any]] = Field(default_factory=list)
    long_term: Dict[str, Any] = Field(default_factory=dict)

class Agent(BaseModel):
    """Base Agent class for all agents in the system."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    memory: AgentMemory = Field(default_factory=AgentMemory)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task with the given context."""
        logger.info(f"Agent {self.name} executing task: {task}")
        
        # Default implementation - override in subclasses
        return {
            "status": "not_implemented",
            "message": "This agent has no execution capability.",
            "agent_id": str(self.id),
            "task": task
        }
    
    def learn(self, data: Any) -> None:
        """Update agent's knowledge based on new data."""
        logger.info(f"Agent {self.name} learning from new data")
        
        # Store in short-term memory
        self.memory.short_term.append({
            "timestamp": time.time(),
            "data": data
        })
        
        self.updated_at = time.time()
    
    def reflect(self) -> Dict[str, Any]:
        """Agent self-reflection to improve performance."""
        logger.info(f"Agent {self.name} reflecting on past actions")
        
        # Default implementation
        return {
            "insights": [],
            "improvements": []
        }
