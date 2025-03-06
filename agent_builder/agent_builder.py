import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field

import numpy as np

# Import model providers
from .model_providers import ModelProvider, OpenAIProvider, AnthropicProvider, ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AgentBuilder')

@dataclass
class AgentCapability:
    """Represents a specific capability that an agent can have."""
    name: str
    description: str
    implementation: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)  # Fix: default_factory.list -> default_factory=list
    parameters: Dict[str, Any] = field(default_factory=dict)  # Fix: default_factory.dict -> default_factory=dict
    # Added performance metrics by model
    performance_by_model: Dict[str, float] = field(default_factory=dict)  # Fix: default_factory.dict -> default_factory=dict

@dataclass
class AgentSpecification:
    """Complete specification for an AI agent."""
    name: str
    description: str
    purpose: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    knowledge_domains: List[str] = field(default_factory=list)  # Fix the default_factory.list typo
    interaction_methods: List[str] = field(default_factory=list)
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    # Added recommended models
    recommended_models: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "dependencies": cap.dependencies,
                    "parameters": cap.parameters,
                    "performance_by_model": cap.performance_by_model
                } for cap in self.capabilities
            ],
            "knowledge_domains": self.knowledge_domains,
            "interaction_methods": self.interaction_methods,
            "model_requirements": self.model_requirements,
            "recommended_models": self.recommended_models
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpecification":
        """Create specification from dictionary."""
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap["description"],
                dependencies=cap.get("dependencies", []),
                parameters=cap.get("parameters", {}),
                performance_by_model=cap.get("performance_by_model", {})
            ) for cap in data.get("capabilities", [])
        ]
        
        return cls(
            name=data["name"],
            description=data["description"],
            purpose=data["purpose"],
            capabilities=capabilities,
            knowledge_domains=data.get("knowledge_domains", []),
            interaction_methods=data.get("interaction_methods", []),
            model_requirements=data.get("model_requirements", {}),
            recommended_models=data.get("recommended_models", [])
        )

class AgentBuilder:
    """Main class for building AI agents."""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                anthropic_api_key: Optional[str] = None,
                default_model: str = "gpt-4"):
        """Initialize the Agent Builder with support for multiple model providers."""
        # Initialize model registry
        self.model_registry = ModelRegistry()
        
        # Register OpenAI provider
        openai_provider = OpenAIProvider(api_key=openai_api_key, default_model=default_model)
        self.model_registry.register_provider("openai", openai_provider)
        
        # Register Anthropic provider if key is provided
        if anthropic_api_key:
            anthropic_provider = AnthropicProvider(api_key=anthropic_api_key)
            self.model_registry.register_provider("anthropic", anthropic_provider)
            
        # Choose default provider
        self.default_provider = openai_provider
        self.default_model = default_model
        
        # Initialize capability library
        self.capability_library = {}
        self.load_capability_library()
        
        # Load performance data if available
        self.performance_data_path = os.path.join(os.path.dirname(__file__), "model_performance.json")
        if os.path.exists(self.performance_data_path):
            self.model_registry.load_performance_data(self.performance_data_path)
    
    def load_capability_library(self) -> None:
        """Load pre-defined capabilities from library."""
        # In a real implementation, this might load from a database or file
        self.capability_library = {
            "text_generation": AgentCapability(
                name="text_generation",
                description="Generate human-like text based on prompts",
                dependencies=["language_model"],
                parameters={"max_tokens": 100, "temperature": 0.7},
                performance_by_model={
                    "gpt-4": 0.9,
                    "gpt-3.5-turbo": 0.85,
                    "claude-2": 0.85
                }
            ),
            "knowledge_retrieval": AgentCapability(
                name="knowledge_retrieval",
                description="Retrieve relevant information from a knowledge base",
                dependencies=["vector_database"],
                parameters={"similarity_threshold": 0.8},
                performance_by_model={
                    "gpt-4": 0.8,
                    "gpt-3.5-turbo": 0.75
                }
            ),
            "task_planning": AgentCapability(
                name="task_planning",
                description="Break down complex tasks into manageable steps",
                dependencies=["reasoning_model"],
                parameters={"planning_depth": 3},
                performance_by_model={
                    "gpt-4": 0.85,
                    "claude-2": 0.85,
                    "gpt-3.5-turbo": 0.7
                }
            ),
            "code_generation": AgentCapability(
                name="code_generation", 
                description="Generate code in various programming languages",
                dependencies=["language_model", "code_knowledge"],
                parameters={"language": "python", "complexity": "medium"},
                performance_by_model={
                    "gpt-4": 0.9,
                    "gpt-3.5-turbo": 0.8,
                    "claude-2": 0.85
                }
            ),
            "conversation_management": AgentCapability(
                name="conversation_management",
                description="Manage multi-turn conversations with context tracking",
                dependencies=["memory_system"],
                parameters={"context_window": 10, "importance_threshold": 0.6},
                performance_by_model={
                    "gpt-4": 0.8,
                    "gpt-3.5-turbo": 0.7,
                    "claude-2": 0.75
                }
            )
        }
    
    def _select_model_for_task(self, task_type: str) -> Tuple[str, ModelProvider]:
        """Select the most appropriate model for a given task type."""
        model_name, provider = self.model_registry.select_best_model(task_type)
        logger.info(f"Selected model {model_name} for task type {task_type}")
        return model_name, provider
    
    def create_agent_specification(self, requirements: Dict[str, Any]) -> AgentSpecification:
        """Create an agent specification based on requirements."""
        # For simple cases, directly map requirements
        name = requirements.get("name", "Unnamed Agent")
        description = requirements.get("description", "An AI agent")
        purpose = requirements.get("purpose", "General purpose assistant")
        
        # Extract requested capabilities
        capability_names = requirements.get("capabilities", [])
        capabilities = []
        
        # Track recommended models based on capability performance
        model_scores = {}
        
        for cap_name in capability_names:
            if cap_name in self.capability_library:
                capability = self.capability_library[cap_name]
                capabilities.append(capability)
                
                # Score models based on capability performance
                for model, score in capability.performance_by_model.items():
                    model_scores[model] = model_scores.get(model, 0) + score
            else:
                # Use LLM to define new capability
                model_name, provider = self._select_model_for_task("capability_definition")
                new_cap = self._generate_capability_definition(cap_name, requirements, provider)
                capabilities.append(new_cap)
                self.capability_library[cap_name] = new_cap
        
        # Sort models by score and get top 3
        recommended_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)[:3]
        
        return AgentSpecification(
            name=name,
            description=description,
            purpose=purpose,
            capabilities=capabilities,
            knowledge_domains=requirements.get("knowledge_domains", []),
            interaction_methods=requirements.get("interaction_methods", ["text"]),
            model_requirements=requirements.get("model_requirements", {"type": "language_model"}),
            recommended_models=recommended_models
        )
    
    def _generate_capability_definition(self, capability_name: str, context: Dict[str, Any], 
                                      provider: ModelProvider) -> AgentCapability:
        """Use LLM to generate a definition for a new capability."""
        prompt = f"""
        Define a new AI agent capability with the name '{capability_name}'.
        
        Context about the agent being built:
        - Purpose: {context.get('purpose', 'General assistant')}
        - Description: {context.get('description', 'An AI agent')}
        
        Please provide:
        1. A clear description of what this capability does
        2. Any dependencies required for this capability
        3. Parameters that can be configured for this capability
        
        Format the response as a JSON object with fields: description, dependencies, parameters.
        """
        
        try:
            system_message = "You are an AI capability designer."
            content, metadata = provider.generate_completion(
                prompt=prompt,
                system_message=system_message,
                max_tokens=500,
                temperature=0.2
            )
            
            # Extract JSON from response
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return AgentCapability(
                    name=capability_name,
                    description=data.get("description", f"Capability for {capability_name}"),
                    dependencies=data.get("dependencies", []),
                    parameters=data.get("parameters", {}),
                    performance_by_model={provider.default_model: 0.7}  # Initial performance estimate
                )
            else:
                logger.error("Could not extract JSON from LLM response")
                return AgentCapability(
                    name=capability_name,
                    description=f"Auto-generated capability for {capability_name}"
                )
                
        except Exception as e:
            logger.error(f"Error generating capability definition: {e}")
            return AgentCapability(
                name=capability_name,
                description=f"Generic capability for {capability_name}"
            )
    
    def analyze_requirements(self, description: str) -> Dict[str, Any]:
        """Use NLP to extract agent requirements from a textual description."""
        # Select model for requirement analysis
        model_name, provider = self._select_model_for_task("requirement_analysis")
        
        prompt = f"""
        Analyze the following description of an AI agent and extract key requirements.
        
        Description: {description}
        
        Extract and format the following information as JSON:
        1. name: A suitable name for the agent
        2. purpose: The main purpose or goal of the agent
        3. capabilities: List of specific capabilities the agent should have
        4. knowledge_domains: Knowledge areas the agent needs to access
        5. interaction_methods: Ways the agent will interact (text, voice, etc.)
        6. model_requirements: Any specific model requirements
        
        Format as a valid JSON object.
        """
        
        try:
            system_message = "You are an AI requirement analyzer."
            content, metadata = provider.generate_completion(
                prompt=prompt,
                system_message=system_message,
                max_tokens=1000,
                temperature=0.2
            )
            
            # Extract JSON from response
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.error("Could not extract JSON from LLM response")
                return {"error": "Could not parse requirements"}
                
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}")
            return {"error": str(e)}
    
    def generate_implementation_plan(self, spec: AgentSpecification) -> Dict[str, Any]:
        """Generate an implementation plan for the specified agent."""
        # Select model for implementation planning
        model_name, provider = self._select_model_for_task("implementation_planning")
        
        spec_json = json.dumps(spec.to_dict(), indent=2)
        
        prompt = f"""
        Create a detailed implementation plan for the following AI agent specification:
        
        {spec_json}
        
        Your plan should include:
        1. architecture: Overall system architecture
        2. components: Key components to implement
        3. integration: How components will be integrated
        4. data_requirements: Data needed for training/operation
        5. evaluation: How to evaluate the agent's performance
        6. implementation_steps: Step-by-step implementation guide
        7. challenges: Potential challenges and mitigations
        
        Format as a valid JSON object.
        """
        
        try:
            system_message = "You are an AI implementation planner."
            content, metadata = provider.generate_completion(
                prompt=prompt,
                system_message=system_message,
                max_tokens=2000,
                temperature=0.2
            )
            
            # Extract JSON from response
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                
                # Track model performance for this task
                task_success = 1.0 if len(plan_data.keys()) >= 5 else 0.7
                self.model_registry.update_model_performance(
                    "openai" if isinstance(provider, OpenAIProvider) else "anthropic",
                    model_name, 
                    "implementation_planning",
                    task_success
                )
                
                # Save updated performance data
                self.model_registry.save_performance_data(self.performance_data_path)
                
                return plan_data
            else:
                logger.error("Could not extract JSON from LLM response")
                return {"error": "Could not generate implementation plan"}
                
        except Exception as e:
            logger.error(f"Error generating implementation plan: {e}")
            return {"error": str(e)}
    
    def generate_code_skeleton(self, implementation_plan: Dict[str, Any]) -> Dict[str, str]:
        """Generate code skeletons based on the implementation plan."""
        # Select model for code generation
        model_name, provider = self._select_model_for_task("code_generation")
        
        plan_json = json.dumps(implementation_plan, indent=2)
        
        prompt = f"""
        Generate code skeletons for the key components in this implementation plan:
        
        {plan_json}
        
        Create Python code skeletons for:
        1. Main agent class
        2. At least one capability implementation
        3. Integration skeleton
        
        Format your response as a valid JSON where keys are filenames and values are the corresponding code.
        For example: {{"agent.py": "class Agent:\\n    ...", "capabilities.py": "class Capability:\\n    ..."}}
        """
        
        try:
            system_message = "You are an AI code generator."
            content, metadata = provider.generate_completion(
                prompt=prompt, 
                system_message=system_message,
                max_tokens=3000,
                temperature=0.2
            )
            
            # Extract JSON from response
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                code_data = json.loads(json_match.group(0))
                
                # Track model performance for this task
                task_success = 1.0 if len(code_data.keys()) >= 2 else 0.6
                self.model_registry.update_model_performance(
                    "openai" if isinstance(provider, OpenAIProvider) else "anthropic",
                    model_name, 
                    "code_generation",
                    task_success
                )
                
                # Save updated performance data
                self.model_registry.save_performance_data(self.performance_data_path)
                
                return code_data
            else:
                logger.error("Could not extract JSON from LLM response")
                return {"error": "Could not generate code skeletons"}
                
        except Exception as e:
            logger.error(f"Error generating code skeletons: {e}")
            return {"error": str(e)}
    
    def save_agent_specification(self, spec: AgentSpecification, filepath: str) -> bool:
        """Save agent specification to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(spec.to_dict(), f, indent=2)
            logger.info(f"Saved agent specification to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving specification: {e}")
            return False
    
    def load_agent_specification(self, filepath: str) -> Optional[AgentSpecification]:
        """Load agent specification from a file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            spec = AgentSpecification.from_dict(data)
            logger.info(f"Loaded agent specification from {filepath}")
            return spec
        except Exception as e:
            logger.error(f"Error loading specification: {e}")
            return None

    def evaluate_agent_design(self, spec: AgentSpecification) -> Dict[str, Any]:
        """Evaluate an agent design for completeness, coherence, and feasibility."""
        evaluation = {
            "completeness": 0.0,
            "coherence": 0.0,
            "feasibility": 0.0,
            "model_fit": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check completeness
        completeness_score = 0.0
        required_fields = ["name", "description", "purpose", "capabilities", 
                          "knowledge_domains", "interaction_methods"]
        
        for field in required_fields:
            value = getattr(spec, field)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    completeness_score += 1.0
                elif not isinstance(value, list) and value:
                    completeness_score += 1.0
        
        evaluation["completeness"] = completeness_score / len(required_fields)
        
        # Check capability coherence
        if spec.capabilities:
            # Check if capabilities align with purpose
            model_name, provider = self._select_model_for_task("coherence_evaluation")
            coherence_score = self._evaluate_capability_coherence(spec, provider)
            evaluation["coherence"] = coherence_score
        
        # Estimate feasibility
        required_deps = set()
        for cap in spec.capabilities:
            required_deps.update(cap.dependencies)
        
        if required_deps:
            # Simplified feasibility check
            common_deps = ["language_model", "vector_database", "memory_system", "reasoning_model"]
            exotic_deps = [dep for dep in required_deps if dep not in common_deps]
            if exotic_deps:
                evaluation["feasibility"] = 0.5
                evaluation["issues"].append(f"Uncommon dependencies required: {', '.join(exotic_deps)}")
                evaluation["recommendations"].append("Consider replacing exotic dependencies with more common alternatives")
            else:
                evaluation["feasibility"] = 0.9
        else:
            evaluation["feasibility"] = 1.0
            
        # Calculate model fit - how well do recommended models align with capabilities
        if spec.recommended_models and spec.capabilities:
            model_fit_scores = []
            for model in spec.recommended_models:
                capability_scores = []
                for cap in spec.capabilities:
                    score = cap.performance_by_model.get(model, 0.5)
                    capability_scores.append(score)
                
                if capability_scores:
                    model_fit_scores.append(sum(capability_scores) / len(capability_scores))
            
            if model_fit_scores:
                evaluation["model_fit"] = sum(model_fit_scores) / len(model_fit_scores)
            else:
                evaluation["model_fit"] = 0.5
        else:
            evaluation["model_fit"] = 0.5
        
        # Overall score
        evaluation["overall_score"] = (evaluation["completeness"] + 
                                      evaluation["coherence"] + 
                                      evaluation["feasibility"] +
                                      evaluation["model_fit"]) / 4
        
        return evaluation
    
    def _evaluate_capability_coherence(self, spec: AgentSpecification, provider: ModelProvider) -> float:
        """Use LLM to evaluate coherence between capabilities and purpose."""
        spec_json = json.dumps(spec.to_dict(), indent=2)
        
        prompt = f"""
        Evaluate the coherence between the agent's purpose and its capabilities:
        
        {spec_json}
        
        On a scale of 0.0 to 1.0, how well do the capabilities support the stated purpose?
        Provide just a single number as your response.
        """
        
        try:
            system_message = "You are an AI design evaluator."
            content, metadata = provider.generate_completion(
                prompt=prompt,
                system_message=system_message,
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract number from response
            number_match = re.search(r'(\d+\.\d+|\d+)', content)
            if number_match:
                score = float(number_match.group(0))
                return min(max(score, 0.0), 1.0)  # Ensure in range [0, 1]
            else:
                logger.error("Could not extract coherence score")
                return 0.5  # Default to middle value
                
        except Exception as e:
            logger.error(f"Error evaluating capability coherence: {e}")
            return 0.5  # Default to middle value

    def optimize_model_selection(self, spec: AgentSpecification, budget_constraint: float = None) -> Dict[str, str]:
        """Optimize model selection for each capability based on performance and cost."""
        model_assignments = {}
        
        if not spec.capabilities:
            return model_assignments
            
        # If we have a budget constraint, try to optimize within budget
        if budget_constraint:
            # Simplified cost model (in arbitrary units)
            model_costs = {
                "gpt-4": 10.0,
                "gpt-3.5-turbo": 1.0,
                "claude-2": 8.0,
                "claude-instant": 2.0
            }
            
            # Sort capabilities by importance (using dependencies as a proxy for importance)
            capabilities_by_importance = sorted(
                spec.capabilities, 
                key=lambda cap: len(cap.dependencies), 
                reverse=True
            )
            
            remaining_budget = budget_constraint
            
            # Assign models starting with most important capabilities
            for cap in capabilities_by_importance:
                # Find best model within remaining budget
                best_score = -1
                best_model = None
                
                for model, score in cap.performance_by_model.items():
                    cost = model_costs.get(model, 5.0)  # Default cost for unknown models
                    if cost <= remaining_budget and score > best_score:
                        best_score = score
                        best_model = model
                
                if best_model:
                    model_assignments[cap.name] = best_model
                    remaining_budget -= model_costs.get(best_model, 5.0)
                else:
                    # If no affordable model, use cheapest option
                    cheapest_model = min(
                        model_costs.keys(), 
                        key=lambda m: model_costs.get(m, float('inf'))
                    )
                    model_assignments[cap.name] = cheapest_model
        
        # Without budget constraints or if budget optimization failed, just use best model for each capability
        for cap in spec.capabilities:
            if cap.name not in model_assignments and cap.performance_by_model:
                best_model = max(
                    cap.performance_by_model.items(), 
                    key=lambda item: item[1]
                )[0]
                model_assignments[cap.name] = best_model
        
        return model_assignments


# Example usage
def example_usage():
    # Initialize agent builder with both OpenAI and Anthropic support
    builder = AgentBuilder(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_model="gpt-4"
    )
    
    # Analyze requirements from a description
    description = """
    I need an AI agent that can help researchers analyze scientific papers.
    It should be able to extract key findings, compare methodologies across papers,
    and suggest related research. It needs to understand scientific terminology,
    especially in the field of machine learning and AI. The agent should be able
    to create summaries at different levels of technical depth depending on the user.
    """
    
    requirements = builder.analyze_requirements(description)
    
    # Create specification
    spec = builder.create_agent_specification(requirements)
    
    # Evaluate the design
    evaluation = builder.evaluate_agent_design(spec)
    
    # Optimize model selection based on capabilities
    model_assignments = builder.optimize_model_selection(spec)
    
    # Generate implementation plan
    plan = builder.generate_implementation_plan(spec)
    
    # Generate code skeletons
    code_skeletons = builder.generate_code_skeleton(plan)
    
    # Save specification
    builder.save_agent_specification(spec, "research_assistant_spec.json")
    
    return {
        "specification": spec.to_dict(),
        "evaluation": evaluation,
        "model_assignments": model_assignments,
        "implementation_plan": plan,
        "code_skeletons": code_skeletons
    }

if __name__ == "__main__":
    result = example_usage()
    print(json.dumps(result, indent=2))