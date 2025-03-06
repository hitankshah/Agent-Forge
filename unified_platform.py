"""
Unified Platform for Agent Management
Provides integration with multiple model providers and database backends.
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Optional, Union
import random
import importlib
import sys
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedPlatform")

class UnifiedPlatform:
    """
    Unified platform for agent creation, management, and deployment.
    Integrates with multiple model providers and database backends.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified platform.
        
        Args:
            config_path: Path to the configuration file. If None, default config is used.
        """
        self.start_time = time.time()
        self.registry = {
            "agents": {},
            "deployments": {},
            "models": {},
            "capabilities": {},
        }
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Ensure data directories exist
        os.makedirs(self.config["data_directory"], exist_ok=True)
        os.makedirs(os.path.join(self.config["data_directory"], "agents"), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_directory"], "deployments"), exist_ok=True)
        
        # Initialize components
        self.components = {}
        self._initialize_components()
        
        # Load existing agents and deployments
        self._load_registry()
        
        logger.info("Unified Platform initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use default.
        
        Args:
            config_path: Path to configuration file.
        
        Returns:
            Dict containing configuration.
        """
        default_config = {
            "data_directory": "data",
            "version": "1.0.0",
            "default_database": "sqlite",
            "database_config": {
                "sqlite": {
                    "db_path": "data/unified_platform.db"
                },
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "user": "postgres",
                    "password": "",
                    "database": "agent_platform"
                },
                "supabase": {
                    "url": "",
                    "key": "",
                    "table": "embeddings"
                }
            },
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                "cohere": os.environ.get("COHERE_API_KEY", "")
            },
            "default_models": {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-2",
                "local": "gemma-7b"
            },
            "open_source_models": {
                "gemma": {
                    "path": "google/gemma-7b",
                    "requires_key": False
                },
                "llama2": {
                    "path": "meta-llama/Llama-2-7b-chat-hf",
                    "requires_key": True
                },
                "mistral": {
                    "path": "mistralai/Mistral-7B-v0.1",
                    "requires_key": False
                }
            },
            "vector_database": {
                "type": "chromadb",
                "path": "data/vector_db"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with default, keeping user settings
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize platform components."""
        # Initialize database
        self._initialize_database()
        
        # Initialize model providers
        self._initialize_model_providers()
        
        # Initialize agent builder
        self._initialize_agent_builder()
    
    def _initialize_database(self):
        """Initialize the database connection."""
        db_type = self.config.get("default_database", "sqlite")
        db_config = self.config.get("database_config", {}).get(db_type, {})
        
        logger.info(f"Initializing {db_type} database")
        
        # Simple mock database for simulation
        self.components["database"] = {
            "type": db_type,
            "config": db_config,
            "connection": None  # Would be actual connection in real implementation
        }
        
        # Initialize vector database if configured
        vector_db_config = self.config.get("vector_database", {})
        vector_db_type = vector_db_config.get("type", "chromadb")
        
        # Mock vector database
        vector_db = {
            "type": vector_db_type,
            "connection": None,  # Would be actual connection in real implementation
            "collections": {}
        }
        
        self.components["database"]["vector_db"] = vector_db
        
        logger.info(f"Initialized {db_type} database and {vector_db_type} vector database")
    
    def _initialize_model_providers(self):
        """Initialize model providers."""
        # Dictionary to store model providers
        model_providers = {}
        
        # Check for API keys
        api_keys = self.config.get("api_keys", {})
        
        # OpenAI
        if api_keys.get("openai"):
            model_providers["openai"] = {
                "available": True,
                "models": ["gpt-4", "gpt-4o", "gpt-3.5-turbo"],
                "default_model": self.config.get("default_models", {}).get("openai", "gpt-3.5-turbo")
            }
        
        # Anthropic
        if api_keys.get("anthropic"):
            model_providers["anthropic"] = {
                "available": True,
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-2"],
                "default_model": self.config.get("default_models", {}).get("anthropic", "claude-2")
            }
        
        # Cohere
        if api_keys.get("cohere"):
            model_providers["cohere"] = {
                "available": True,
                "models": ["command", "command-light"],
                "default_model": self.config.get("default_models", {}).get("cohere", "command")
            }
        
        # Open source models
        open_source_models = self.config.get("open_source_models", {})
        os_models = []
        
        for model_name, model_info in open_source_models.items():
            os_models.append({
                "name": model_name,
                "path": model_info.get("path"),
                "requires_key": model_info.get("requires_key", False)
            })
        
        model_providers["open_source"] = {
            "available": True,
            "models": os_models,
            "default_model": self.config.get("default_models", {}).get("local", "gemma-7b")
        }
        
        self.components["model_providers"] = model_providers
        logger.info("Model providers initialized")
    
    def _initialize_agent_builder(self):
        """Initialize the agent builder component."""
        # Simple mock agent builder
        self.components["agent_builder"] = {
            "builder": AgentBuilder(self.components["model_providers"])
        }
        logger.info("Agent builder initialized")
    
    def _load_registry(self):
        """Load existing agents and deployments from disk."""
        # Load agents
        agents_dir = os.path.join(self.config["data_directory"], "agents")
        if os.path.exists(agents_dir):
            for filename in os.listdir(agents_dir):
                if filename.endswith(".json"):
                    agent_path = os.path.join(agents_dir, filename)
                    try:
                        with open(agent_path, 'r') as f:
                            agent_data = json.load(f)
                            self.registry["agents"][agent_data["id"]] = agent_data
                    except Exception as e:
                        logger.error(f"Error loading agent {filename}: {e}")
        
        # Load deployments
        deployments_dir = os.path.join(self.config["data_directory"], "deployments")
        if os.path.exists(deployments_dir):
            for filename in os.listdir(deployments_dir):
                if filename.endswith(".json"):
                    deployment_path = os.path.join(deployments_dir, filename)
                    try:
                        with open(deployment_path, 'r') as f:
                            deployment_data = json.load(f)
                            self.registry["deployments"][deployment_data["id"]] = deployment_data
                    except Exception as e:
                        logger.error(f"Error loading deployment {filename}: {e}")
        
        # Load samples if no agents exist
        if not self.registry["agents"]:
            self._load_sample_agents()
        
        logger.info(f"Loaded {len(self.registry['agents'])} agents and {len(self.registry['deployments'])} deployments")
    
    def _load_sample_agents(self):
        """Load sample agents if no agents exist."""
        sample_agents = [
            {
                "id": str(uuid.uuid4()),
                "name": "Research Assistant",
                "description": "AI agent that helps with academic research and paper analysis",
                "specification": {
                    "purpose": "Help researchers analyze academic papers and extract key information",
                    "capabilities": [
                        {
                            "name": "Paper Analysis", 
                            "description": "Extract key findings from research papers",
                            "dependencies": ["document_understanding", "summary"],
                            "parameters": {"max_papers": 5}
                        },
                        {
                            "name": "Citation Generation",
                            "description": "Generate citations for papers in various formats",
                            "dependencies": ["metadata_extraction"],
                            "parameters": {"formats": ["APA", "MLA", "Chicago"]}
                        }
                    ],
                    "knowledge_domains": ["Academic Research", "Scientific Papers", "Citations"],
                    "recommended_models": ["gpt-4", "claude-3-opus"]
                },
                "created_at": datetime.datetime.now().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Personal Assistant",
                "description": "Everyday assistant for scheduling, reminders, and information lookup",
                "specification": {
                    "purpose": "Help users manage their day-to-day tasks and information needs",
                    "capabilities": [
                        {
                            "name": "Calendar Management", 
                            "description": "Create, update and track calendar events",
                            "dependencies": ["date_time_understanding"],
                            "parameters": {"calendar_sync": True}
                        },
                        {
                            "name": "Information Lookup",
                            "description": "Search for information online",
                            "dependencies": ["web_search"],
                            "parameters": {"search_engine": "default"}
                        }
                    ],
                    "knowledge_domains": ["General Knowledge", "Scheduling", "Task Management"],
                    "recommended_models": ["gpt-3.5-turbo", "claude-2", "gemma-7b"]
                },
                "created_at": datetime.datetime.now().isoformat()
            }
        ]
        
        # Add sample agents to registry
        for agent in sample_agents:
            self.registry["agents"][agent["id"]] = agent
            
            # Save to disk
            agent_path = os.path.join(self.config["data_directory"], "agents", f"{agent['id']}.json")
            try:
                with open(agent_path, 'w') as f:
                    json.dump(agent, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving sample agent {agent['id']}: {e}")
        
        logger.info(f"Loaded {len(sample_agents)} sample agents")
    
    def create_agent(self, description: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new agent from a description.
        
        Args:
            description: Description of the agent.
            name: Optional name for the agent. If None, a name will be generated.
        
        Returns:
            Dict containing the created agent information.
        """
        try:
            # Generate a unique ID
            agent_id = str(uuid.uuid4())
            
            # Generate name if needed
            if not name:
                name = f"Agent {len(self.registry['agents']) + 1}"
            
            # In a real implementation, this would use an LLM to generate the agent specification
            agent_spec = self._generate_agent_specification(description)
            
            # Ensure the specification includes the agent's name
            agent_spec['name'] = name
            
            # Create agent object
            agent = {
                "id": agent_id,
                "name": name,
                "description": description,
                "specification": agent_spec,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Evaluate agent (in a real implementation, would use more sophisticated metrics)
            evaluation = {
                "completeness": random.uniform(0.7, 0.95),
                "coherence": random.uniform(0.7, 0.97),
                "feasibility": random.uniform(0.6, 0.95),
                "model_fit": random.uniform(0.75, 0.98),
                "overall_score": random.uniform(0.7, 0.95),
                "issues": [
                    "Some capabilities may require additional refinement"
                ] if random.random() > 0.7 else [],
                "recommendations": [
                    "Consider adding more specific examples for better performance"
                ] if random.random() > 0.5 else []
            }
            
            # Add to registry
            self.registry["agents"][agent_id] = agent
            
            # Save to disk
            agents_dir = os.path.join(self.config["data_directory"], "agents")
            os.makedirs(agents_dir, exist_ok=True)
            with open(os.path.join(agents_dir, f"{agent_id}.json"), 'w') as f:
                json.dump(agent, f, indent=2)
            
            logger.info(f"Created agent: {name} ({agent_id})")
            return {
                "success": True,
                "agent_id": agent_id,
                "specification": agent_spec,
                "evaluation": evaluation
            }
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return {
                "success": False,
                "errors": [str(e)]
            }
    
    def _generate_agent_specification(self, description: str) -> Dict[str, Any]:
        """
        Generate agent specification from description.
        In a real implementation, this would use an LLM.
        
        Args:
            description: Description of the agent.
            
        Returns:
            Dict containing the agent specification.
        """
        # Extract keywords to inform the mock agent capabilities
        keywords = set(word.lower() for word in description.replace(".", "").replace(",", "").split())
        
        # Define some templates based on keywords
        templates = {
            "research": {
                "purpose": "Assist with research and information gathering",
                "capabilities": [
                    {
                        "name": "Literature Review",
                        "description": "Survey and summarize existing research papers and articles",
                        "dependencies": ["document_understanding", "summary"],
                        "parameters": {"depth": "comprehensive"}
                    },
                    {
                        "name": "Citation Management",
                        "description": "Organize and format citations",
                        "dependencies": ["metadata_extraction"],
                        "parameters": {"style": "configurable"}
                    }
                ],
                "knowledge_domains": ["Academic Research", "Information Analysis"],
                "recommended_models": ["gpt-4", "claude-3-opus", "gemma-7b"]
            },
            "write": {
                "purpose": "Generate and refine written content",
                "capabilities": [
                    {
                        "name": "Content Generation",
                        "description": "Create written content based on provided topic or outline",
                        "dependencies": ["language_generation", "creativity"],
                        "parameters": {"tone": "adaptable", "length": "variable"}
                    },
                    {
                        "name": "Editing and Proofreading",
                        "description": "Improve existing content for clarity, grammar, and style",
                        "dependencies": ["grammar_checking", "style_analysis"],
                        "parameters": {"strictness": "adjustable"}
                    }
                ],
                "knowledge_domains": ["Writing", "Editing", "Content Creation"],
                "recommended_models": ["gpt-4", "claude-3-opus", "llama-2-7b"]
            },
            "analyze": {
                "purpose": "Analyze data and extract insights",
                "capabilities": [
                    {
                        "name": "Data Analysis",
                        "description": "Process and interpret data to identify patterns and trends",
                        "dependencies": ["statistical_analysis", "data_visualization"],
                        "parameters": {"data_format": "flexible", "output_type": "configurable"}
                    },
                    {
                        "name": "Insight Generation",
                        "description": "Generate actionable insights from analysis",
                        "dependencies": ["critical_thinking", "domain_knowledge"],
                        "parameters": {"depth": "adjustable", "format": "customizable"}
                    }
                ],
                "knowledge_domains": ["Data Analysis", "Statistics", "Insight Generation"],
                "recommended_models": ["gpt-4", "claude-3-opus", "mistral-7b"]
            },
            "chat": {
                "purpose": "Engage in helpful conversation",
                "capabilities": [
                    {
                        "name": "Conversational Response",
                        "description": "Generate contextually appropriate and helpful responses",
                        "dependencies": ["context_awareness", "natural_language_understanding"],
                        "parameters": {"style": "adaptive", "memory": "session"}
                    },
                    {
                        "name": "Personality",
                        "description": "Maintain consistent tone and personality",
                        "dependencies": ["consistency", "characterization"],
                        "parameters": {"persona": "configurable"}
                    }
                ],
                "knowledge_domains": ["Conversation", "Communication"],
                "recommended_models": ["gpt-3.5-turbo", "claude-2", "gemma-7b"]
            }
        }
        
        # Find template with most keyword matches
        best_match = None
        best_match_count = 0
        for template_key, template in templates.items():
            if template_key in keywords:
                if not best_match or template_key in description.lower():
                    best_match = template
                    best_match_count += 1
        
        # If no good match, use chat template
        if not best_match:
            best_match = templates["chat"]
        
        # Apply some variations based on specific keywords
        purpose = best_match["purpose"]
        capabilities = best_match["capabilities"].copy()
        knowledge_domains = best_match["knowledge_domains"].copy()
        recommended_models = best_match["recommended_models"].copy()
        
        # Modify purpose based on description keywords
        if "research" in keywords:
            purpose = f"{purpose} in research contexts"
            if "Research" not in knowledge_domains:
                knowledge_domains.append("Research")
        
        if "data" in keywords:
            if "Data" not in knowledge_domains:
                knowledge_domains.append("Data Analysis")
            
        if "code" in keywords or "programming" in keywords:
            capabilities.append({
                "name": "Code Generation",
                "description": "Generate and explain code in multiple programming languages",
                "dependencies": ["programming_knowledge"],
                "parameters": {"languages": ["Python", "JavaScript", "Other"]}
            })
            knowledge_domains.append("Programming")
            
        if "summarize" in keywords or "summary" in keywords:
            capabilities.append({
                "name": "Text Summarization",
                "description": "Create concise summaries of longer documents",
                "dependencies": ["content_understanding"],
                "parameters": {"length": "adjustable", "focus": "key points"}
            })
        
        # Return the specification
        return {
            "purpose": purpose,
            "capabilities": capabilities,
            "knowledge_domains": knowledge_domains,
            "recommended_models": recommended_models
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of available agents.
        
        Returns:
            List of agent objects.
        """
        return list(self.registry["agents"].values())
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve.
            
        Returns:
            Agent object if found, None otherwise.
        """
        return self.registry["agents"].get(agent_id)
    
    def deploy_agent(self, agent_id: str, deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy an agent.
        
        Args:
            agent_id: ID of the agent to deploy.
            deployment_config: Optional deployment configuration.
            
        Returns:
            Dict containing deployment information.
        """
        try:
            agent = self.get_agent(agent_id)
            if not agent:
                return {"success": False, "errors": [f"Agent with ID {agent_id} not found"]}
            
            # Default config if none provided
            if not deployment_config:
                deployment_config = {
                    "environment": "production",
                    "scaling": "auto",
                    "memory": 512,  # MB
                    "timeout": 60,   # seconds
                    "created_at": datetime.datetime.now().isoformat(),
                    "endpoint": f"api/agents/{agent_id}/query"
                }
            
            # Generate model assignments for capabilities
            model_assignments = {}
            for capability in agent["specification"]["capabilities"]:
                # Randomly select one of the recommended models
                recommended_models = agent["specification"].get("recommended_models", ["gpt-3.5-turbo"])
                model_assignments[capability["name"]] = random.choice(recommended_models)
            
            deployment_config["model_assignments"] = model_assignments
            
            # Generate unique ID for deployment
            deployment_id = str(uuid.uuid4())
            
            # Create deployment object
            deployment = {
                "id": deployment_id,
                "agent_id": agent_id,
                "status": "active",
                "config": deployment_config
            }
            
            # Add to registry
            self.registry["deployments"][deployment_id] = deployment
            
            # Save to disk
            deployments_dir = os.path.join(self.config["data_directory"], "deployments")
            os.makedirs(deployments_dir, exist_ok=True)
            with open(os.path.join(deployments_dir, f"{deployment_id}.json"), 'w') as f:
                json.dump(deployment, f, indent=2)
            
            logger.info(f"Deployed agent {agent_id} as {deployment_id}")
            return {
                "success": True,
                "deployment_id": deployment_id,
                "status": "active",
                "config": deployment_config
            }
        except Exception as e:
            logger.error(f"Error deploying agent: {e}")
            return {
                "success": False,
                "errors": [str(e)]
            }
    
    def query_agent(self, agent_id: str, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query an agent.
        
        Args:
            agent_id: ID of the agent to query.
            query: Query string.
            parameters: Optional query parameters.
            
        Returns:
            Dict containing the agent response.
        """
        try:
            agent = self.get_agent(agent_id)
            if not agent:
                return {"success": False, "errors": [f"Agent with ID {agent_id} not found"]}
            
            # Find active deployment for agent
            deployment = None
            for d in self.registry["deployments"].values():
                if d["agent_id"] == agent_id and d["status"] == "active":
                    deployment = d
                    break
            
            if not deployment:
                return {"success": False, "errors": ["No active deployment found for agent"]}
            
            # In a real implementation, this would use models to generate responses
            # based on the agent's capabilities and the query
            
            # Mock response generation - in a real system this would call an LLM
            response = self._generate_response(agent, query)
            
            # Mock metadata
            metadata = {
                "model": next(iter(deployment["config"].get("model_assignments", {}).values()), "gpt-3.5-turbo"),
                "processing_time": random.uniform(0.5, 2.0),
                "usage": {
                    "prompt_tokens": random.randint(100, 300),
                    "completion_tokens": random.randint(50, 500),
                    "total_tokens": random.randint(150, 800)
                }
            }
            
            logger.info(f"Agent {agent_id} queried: {query[:50]}...")
            return {
                "success": True,
                "response": response,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error querying agent: {e}")
            return {
                "success": False, 
                "errors": [str(e)]
            }
    
    def _generate_response(self, agent: Dict[str, Any], query: str) -> str:
        """
        Generate response for a query based on agent capabilities.
        
        Args:
            agent: Agent object.
            query: Query string.
            
        Returns:
            String response.
        """
        # Simple response templates based on agent type
        templates = {
            "research": [
                "Based on my research capabilities, I can tell you that {query_topic} involves several key considerations.",
                "Looking at the academic literature on {query_topic}, there are multiple perspectives to consider.",
                "My analysis of {query_topic} draws on several scholarly sources which suggest that...",
                "From a research perspective, {query_topic} presents interesting challenges that scholars have approached in different ways."
            ],
            "write": [
                "Here's a draft about {query_topic} that you might find useful.",
                "I've prepared some content about {query_topic} based on your request.",
                "When writing about {query_topic}, it's important to consider the following key points...",
                "Here's my written response regarding {query_topic}."
            ],
            "analyze": [
                "My analysis of {query_topic} reveals several important patterns.",
                "Looking at the data related to {query_topic}, I can identify the following insights.",
                "From an analytical perspective, {query_topic} demonstrates these characteristics.",
                "The key metrics for {query_topic} indicate several important trends."
            ],
            "chat": [
                "I'd be happy to discuss {query_topic} with you. What specific aspects are you interested in?",
                "Thanks for asking about {query_topic}. Here's what I know.",
                "That's an interesting question about {query_topic}. Let me share some thoughts.",
                "I'm glad you asked about {query_topic}. Here's what I can tell you."
            ]
        }
        
        # Determine agent type based on capabilities
        agent_type = "chat"  # default
        capabilities = [cap["name"].lower() for cap in agent["specification"]["capabilities"]]
        
        if any(kw in cap for cap in capabilities for kw in ["research", "literature", "academic"]):
            agent_type = "research"
        elif any(kw in cap for cap in capabilities for kw in ["write", "content", "editing"]):
            agent_type = "write"
        elif any(kw in cap for cap in capabilities for kw in ["analy", "data", "insight"]):
            agent_type = "analyze"
        
        # Extract topic from query (simple approach)
        query_topic = query.strip("?!.")
        if len(query_topic) > 30:
            query_topic = query_topic[:30] + "..."
        
        # Select template
        template = random.choice(templates[agent_type])
        intro = template.format(query_topic=query_topic)
        
        # Generate a more detailed response based on agent capabilities
        details = []
        for capability in agent["specification"]["capabilities"]:
            if random.random() > 0.5:  # Only use some capabilities
                details.append(f"Using my {capability['name']} capability, I can provide additional insights about this topic.")
        
        # Add a conclusion
        conclusions = [
            "Please let me know if you'd like more specific information about any aspect of this.",
            "I hope this information is helpful. Feel free to ask if you need clarification.",
            "I can provide more detailed analysis if needed - just let me know what interests you.",
            "Let me know if you'd like me to explore any particular aspect in more depth."
        ]
        conclusion = random.choice(conclusions)
        
        # Assemble the complete response
        response_parts = [intro]
        if details:
            response_parts.append("\n\n" + "\n".join(details))
        response_parts.append("\n\n" + conclusion)
        
        return "".join(response_parts)

    def get_status(self) -> Dict[str, Any]:
        """
        Get platform status information.
        
        Returns:
            Dict with status information.
        """
        # Get uptime
        uptime_seconds = int(time.time() - self.start_time)
        
        # Get component status
        components = {
            "database": "database" in self.components,
            "model_providers": "model_providers" in self.components,
            "agent_builder": "agent_builder" in self.components
        }
        
        # Get registry counts
        registry_counts = {
            "agents_count": len(self.registry["agents"]),
            "deployments_count": len(self.registry["deployments"]),
            "models_count": len(self.registry["models"]),
            "capabilities_count": sum(len(agent["specification"]["capabilities"]) 
                                   for agent in self.registry["agents"].values())
        }
        
        # Mock system metrics
        metrics = {
            "cpu": random.uniform(10, 40),
            "memory": random.uniform(20, 60),
            "disk": random.uniform(30, 70),
            "network": random.uniform(5, 30)
        }
        
        return {
            "version": self.config["version"],
            "uptime_seconds": uptime_seconds,
            "components": components,
            "registry": registry_counts,
            "metrics": metrics
        }
        
    def generate_implementation_plan(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate an implementation plan for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dict containing the implementation plan.
        """
        try:
            agent = self.get_agent(agent_id)
            if not agent:
                return {"success": False, "errors": [f"Agent with ID {agent_id} not found"]}
            
            # Mock implementation plan
            capabilities = agent["specification"]["capabilities"]
            
            steps = []
            for i, capability in enumerate(capabilities):
                steps.append({
                    "step": i + 1,
                    "name": f"Implement {capability['name']}",
                    "description": f"Create the {capability['name']} capability",
                    "estimated_time": f"{random.randint(1, 5)} hours",
                    "dependencies": capability["dependencies"]
                })
            
            implementation_plan = {
                "agent_id": agent_id,
                "agent_name": agent["name"],
                "steps": steps,
                "total_estimated_time": f"{sum(random.randint(1, 5) for _ in capabilities)} hours",
                "recommended_tech_stack": ["Python", "FastAPI", "Streamlit"]
            }
            
            return {
                "success": True,
                "plan": implementation_plan
            }
            
        except Exception as e:
            logger.error(f"Error generating implementation plan: {e}")
            return {
                "success": False,
                "errors": [str(e)]
            }
            
    def generate_code(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate code for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dict containing the generated code.
        """
        try:
            agent = self.get_agent(agent_id)
            if not agent:
                return {"success": False, "errors": [f"Agent with ID {agent_id} not found"]}
            
            # Mock code generation
            capabilities = agent["specification"]["capabilities"]
            
            # Create main agent file
            main_file = f"""
# {agent['name']} Agent
# Generated code for agent {agent_id}

import os
import json
from typing import Dict, Any, List

class {agent['name'].replace(' ', '')}:
    \"\"\"
    {agent['description']}
    \"\"\"
    
    def __init__(self):
        self.name = "{agent['name']}"
        self.capabilities = {json.dumps([cap['name'] for cap in capabilities], indent=4)}
        
    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        \"\"\"Process a user query.\"\"\"
        # Determine which capability to use
        capability = self._select_capability(query)
        
        # Process with the selected capability
        if capability == "default":
            return self._default_response(query)
            
        {self._generate_capability_conditionals(capabilities)}
        
        # Fall back to default
        return self._default_response(query)
    
    def _select_capability(self, query: str) -> str:
        \"\"\"Select the appropriate capability based on the query.\"\"\"
        # In production, this would use NLP to select the best capability
        # For now, just use simple keyword matching
        query_lower = query.lower()
        
        {self._generate_keyword_matching(capabilities)}
        
        return "default"
        
    def _default_response(self, query: str) -> Dict[str, Any]:
        \"\"\"Generate default response.\"\"\"
        return {
            "response": f"I'll help you with {{query}}.",
            "metadata": {
                "capability": "default",
                "confidence": 0.7
            }
        }
"""
            
            # Create additional capability files
            capability_files = {}
            for capability in capabilities:
                cap_name = capability["name"].replace(" ", "_").lower()
                cap_code = f"""
# {capability['name']} capability module
# Part of {agent['name']} Agent

class {cap_name.title().replace('_', '')}:
    \"\"\"
    {capability['description']}
    \"\"\"
    
    def __init__(self):
        self.name = "{capability['name']}"
        self.params = {json.dumps(capability['parameters'], indent=8)}
        
    def process(self, query: str, **kwargs):
        \"\"\"Process a query using this capability.\"\"\"
        # In production, this would implement the actual capability logic
        return {{
            "response": f"Using {capability['name']} capability to address your query about {{query}}",
            "metadata": {{
                "capability": "{capability['name']}",
                "confidence": 0.9
            }}
        }}
"""
                capability_files[f"{cap_name}.py"] = cap_code
            
            # Create requirements file
            requirements = """
# Requirements for {agent_name}
fastapi>=0.95.0
pydantic>=2.0.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
requests>=2.28.0
""".format(agent_name=agent['name'])

            # Create README
            readme = f"""# {agent['name']}

{agent['description']}

## Capabilities

{self._generate_capability_docs(capabilities)}

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the agent:
   ```
   python main.py
   ```

## API

- `POST /query` - Send a query to the agent
  - Request body: `{{"query": "your query here"}}`
  - Response: `{{"response": "agent response", "metadata": {{}}}}`
"""

            # Compile code collection
            code = {
                "main.py": main_file,
                "requirements.txt": requirements,
                "README.md": readme
            }
            
            # Add capability files
            code.update(capability_files)
            
            return {
                "success": True,
                "code": code
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                "success": False,
                "errors": [str(e)]
            }
    
    def _generate_capability_conditionals(self, capabilities):
        """Generate conditional code blocks for capabilities."""
        blocks = []
        for capability in capabilities:
            cap_name = capability["name"].replace(" ", "_").lower()
            blocks.append(f'elif capability == "{capability["name"]}":\n            from {cap_name} import {cap_name.title().replace("_", "")}\n            handler = {cap_name.title().replace("_", "")}()\n            return handler.process(query)')
        
        return "\n        ".join(blocks)
    
    def _generate_keyword_matching(self, capabilities):
        """Generate keyword matching code for capability selection."""
        blocks = []
        for capability in capabilities:
            keywords = capability["name"].lower().split() + capability["description"].lower().split()[:5]
            keywords = list(set([k for k in keywords if len(k) > 3]))
            
            if keywords:
                keyword_checks = " or ".join([f'"{kw}" in query_lower' for kw in keywords[:3]])
                blocks.append(f'if {keyword_checks}:\n            return "{capability["name"]}"')
        
        return "\n        ".join(blocks)
    
    def _generate_capability_docs(self, capabilities):
        """Generate markdown documentation for capabilities."""
        docs = []
        for capability in capabilities:
            params = ", ".join([f"`{k}`: {v}" for k, v in capability["parameters"].items()])
            docs.append(f"- **{capability['name']}**: {capability['description']}\n  - Parameters: {params}")
        
        return "\n\n".join(docs)

class AgentBuilder:
    def __init__(self, model_provider):
        self.model_provider = model_provider

    def analyze_requirements(self, description: str) -> Dict[str, Any]:
        """
        Analyze user requirements and generate agent specification.
        
        Args:
            description: Description of the agent requirements.
            
        Returns:
            Dict containing the agent specification.
        """
        try:
            # Generate prompt for the LLM
            prompt = create_structured_prompt("Analyze the following requirements and provide a JSON specification:", description)
            
            # Get response from the LLM
            response = self.model_provider.generate_text(prompt)
            
            # Attempt to extract JSON from the response
            json_data = extract_json_from_llm_response(response)
            
            if not json_data:
                logger.error(f"Could not extract JSON from LLM response: {response}")
                return {"success": False, "errors": ["Could not extract JSON from LLM response"]}
            
            return {"success": True, "specification": json_data}
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}")
            return {"success": False, "errors": [str(e)]}