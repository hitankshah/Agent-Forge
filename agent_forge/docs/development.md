# Agent Forge Development Guide

This guide provides detailed information about the Agent Forge project structure, components, and how to extend or modify the system to suit your needs.

## Project Structure

```
agent_forge/
├── core/                    # Core agent functionality
│   ├── agent.py             # Base Agent class definition
│   ├── agent_factory.py     # Factory for creating agents
│   └── specialized_agents/  # Specialized agent implementations
│       ├── researcher.py    # Researcher agent implementation
│       └── ...
├── integrations/            # External service integrations
│   ├── database.py          # Vector database integrations
│   ├── embeddings.py        # Embedding generation utilities
│   └── llm.py               # LLM provider integrations
├── ui/                      # User interface
│   └── app.py               # Streamlit UI implementation
├── utils/                   # Utility functions
│   ├── config.py            # Configuration handling
│   └── templates.py         # Prompt template management
├── main.py                  # Application entry point
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Multi-container Docker setup
└── requirements.txt         # Python dependencies
```

## Core Components

### Agent System (`core/agent.py`)

The Agent class is the foundational building block of the system:

- **AgentMemory**: Manages short-term and long-term agent memory
- **Agent**: Base class with core capabilities that all agents inherit:
  - `execute()`: Processes tasks and generates responses
  - `learn()`: Updates agent knowledge based on new data
  - `reflect()`: Enables self-improvement through reflection

**How to modify**: You can extend the Agent class to add new capabilities by overriding these methods or adding new ones:

```python
from agent_forge.core.agent import Agent

class MyCustomAgent(Agent):
    def execute(self, task, context=None):
        # Custom implementation
        return {
            "status": "success",
            "message": f"I processed {task} with my custom logic",
            "agent_id": str(self.id),
            "task": task
        }
    
    def new_capability(self):
        # Add new capabilities
        pass
```

### Agent Factory (`core/agent_factory.py`)

The AgentFactory class manages agent creation and customization:

- **register_agent_type()**: Adds new agent types to the system
- **create_agent()**: Creates an agent with specified parameters
- **generate_agent_from_description()**: Uses LLMs to create an agent from a text description

**How to modify**: Register your custom agent types with the factory:

```python
from agent_forge.core.agent_factory import AgentFactory
from my_agents import MyCustomAgent

# Get factory instance
factory = AgentFactory(config)

# Register your custom agent
factory.register_agent_type("my_custom_agent", MyCustomAgent)
```

## Integration Components

### LLM Providers (`integrations/llm.py`)

Supports multiple LLM providers through a unified interface:

- **BaseLLMProvider**: Abstract base class all providers must implement
- **Available Providers**:
  - **OpenAIProvider**: OpenAI API integration (GPT models)
  - **AnthropicProvider**: Anthropic API integration (Claude models)
  - **GoogleProvider**: Google Gemini models integration
  - **CohereLLMProvider**: Cohere API integration
  - **OllamaProvider**: Local model integration via Ollama

**How to modify**: Add a new LLM provider by extending BaseLLMProvider:

```python
from agent_forge.integrations.llm import BaseLLMProvider

class MyProviderIntegration(BaseLLMProvider):
    def __init__(self, config):
        # Initialize with your provider's configuration
        self.api_key = config.get("api_key")
        # ...
        
    def provider_name(self):
        return "My Provider"
    
    def generate(self, template, inputs):
        # Implementation for generating text
        pass
        
    def generate_with_history(self, messages):
        # Implementation for chat functionality
        pass
```

Then update the LLMProvider class to include your new provider.

### Database Integration (`integrations/database.py`)

Vector database abstraction for storing and retrieving embeddings:

- **BaseVectorStore**: Abstract base class for database providers
- **Available Stores**:
  - **ChromaDBStore**: ChromaDB integration
  - **SupabaseVectorStore**: Supabase with pgvector integration
  - **PineconeVectorStore**: Pinecone vector database integration

**How to modify**: Add a new vector database by extending BaseVectorStore:

```python
from agent_forge.integrations.database import BaseVectorStore

class MyVectorDBStore(BaseVectorStore):
    def __init__(self, config):
        # Initialize with your database configuration
        pass
        
    def store_embedding(self, text, embedding, metadata=None):
        # Store embedding implementation
        pass
        
    def query_similar(self, query_embedding, top_k=5):
        # Similarity search implementation
        pass
        
    # Implement other required methods...
```

### Embeddings Generation (`integrations/embeddings.py`)

Utilities for generating embeddings from text:

- **get_embedding()**: Unified function for generating embeddings from different providers
- Supports OpenAI, Cohere, and Hugging Face embedding models

**How to modify**: Add support for a new embedding provider:

```python
# In embeddings.py
def get_my_provider_embedding(text, model="my-model"):
    # Implementation for your embedding provider
    pass
    
# Update the get_embedding function to include your provider
def get_embedding(text, model="text-embedding-ada-002"):
    # Add to conditionals
    if "my-provider" in model:
        return get_my_provider_embedding(text, model)
    # ... other providers
```

## User Interface (`ui/app.py`)

The UI is built with Streamlit and provides:

- Agent creation and management
- Interactive chat with agents
- Configuration editing
- Memory visualization

**How to modify**: The UI is based on Streamlit - you can modify `app.py` to:

- Add new UI sections or tabs
- Create additional views for specialized agents
- Customize the appearance and layout
- Add new interactive elements

## Configuration System (`utils/config.py`)

Manages application settings through:

- YAML configuration files
- Environment variables
- Default values

**How to modify**: Add new configuration sections by updating:
1. The default configuration dictionary in `load_config()`
2. Adding new environment variable mappings
3. Creating a configuration file with your custom settings

## Prompt Templates (`utils/templates.py`)

Manages reusable prompt templates for agent interactions:

- **DEFAULT_TEMPLATES**: Dictionary of built-in templates
- **load_template()**: Loads templates from files or built-ins

**How to modify**: Add new templates either by:

1. Adding to the DEFAULT_TEMPLATES dictionary:
```python
DEFAULT_TEMPLATES["my_template"] = """
This is my custom template with {placeholder}.
"""
```

2. Creating template files in a "templates" directory:
```
templates/my_template.txt
```

## Docker Integration

The project includes Docker support for containerization:

- **Dockerfile**: Defines the main application container
- **docker-compose.yml**: Sets up the complete environment with databases

**How to modify**: You can customize the Docker setup by:

1. Modifying the Dockerfile to add custom dependencies or configuration
2. Updating docker-compose.yml to add new services or configure existing ones
3. Adding environment-specific compose files (e.g., development, production)

## Database Setup Examples

### Supabase Setup

To use Supabase with pgvector for embedding storage:

1. Create a Supabase project at https://supabase.com
2. Enable the pgvector extension for your project
3. Create a table for embeddings:

```sql
CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  content TEXT NOT NULL,
  embedding VECTOR(1536),
  metadata JSONB DEFAULT '{}'::JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a vector index
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION query_embeddings(
  query_embedding VECTOR,
  match_count INT DEFAULT 5
) 
RETURNS TABLE (id UUID, content TEXT, metadata JSONB, similarity FLOAT) 
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.content,
    embeddings.metadata,
    1 - (embeddings.embedding <=> query_embedding) AS similarity
  FROM embeddings
  ORDER BY embeddings.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

4. Add your Supabase URL and key to your `.env` file:
```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-key
```

5. Update your configuration to use Supabase:
```yaml
database:
  type: supabase
  supabase:
    table_name: embeddings
```

### Pinecone Setup

To use Pinecone for vector storage:

1. Create a Pinecone account and project at https://www.pinecone.io/
2. Create an index with the appropriate dimension (1536 for OpenAI embeddings)
3. Add your Pinecone API key to your `.env` file:
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-environment (e.g., us-west1-gcp)
```

4. Update your configuration:
```yaml
database:
  type: pinecone
  pinecone:
    index_name: agent-forge
    dimension: 1536
```

## Extending with New Agent Types

To create a specialized agent:

1. Create a new file in `core/specialized_agents/`
2. Define a class that inherits from the base Agent
3. Override methods like `execute()` and `reflect()`
4. Register the agent type with the AgentFactory

Example for a custom Coder agent:

```python
# In core/specialized_agents/coder.py
from agent_forge.core.agent import Agent

class CoderAgent(Agent):
    def execute(self, task, context=None):
        # Implement code generation logic
        pass
        
    def reflect(self):
        # Implement reflection on code quality
        pass

# In your application initialization
factory.register_agent_type("coder", CoderAgent)
```

## Common Customization Scenarios

### Adding a New LLM Provider

1. Create a new class that inherits from `BaseLLMProvider`
2. Implement the required methods
3. Update the `LLMProvider._initialize_provider()` method to include your provider

### Customizing Agent Behavior

1. Extend the base Agent class or a specialized agent
2. Override the `execute()` method to implement custom behavior
3. Register your agent type with the AgentFactory

### Adding Vector Database Support

1. Create a new class that inherits from `BaseVectorStore`
2. Implement the required methods for storing and querying embeddings
3. Update the `DatabaseManager._initialize_store()` method to include your database

## Using Custom LLM Models

Agent Forge supports integration with any LLM model that has an API. This is a key feature that differentiates it from other agent frameworks.

### Custom Model API Support

Two API formats are supported:

1. **OpenAI-compatible API** (default): 
   - Used by many model providers like OpenRouter, Together AI, etc.
   - Follows the format with /chat/completions endpoint
   - JSON format with messages array

2. **Raw Completion API**:
   - Simpler API format that just takes a prompt and returns text
   - Used by some simpler model APIs

### Setting Up Custom Models

#### Via Environment Variables

```bash
CUSTOM_MODEL=your_model_name
CUSTOM_BASE_URL=https://your-api-endpoint/v1
CUSTOM_API_KEY=your_api_key
CUSTOM_API_TYPE=openai_compatible  # or raw_completion
```

#### Via Configuration File

```yaml
llm:
  provider: custom
  custom:
    model: your_model_name
    base_url: https://your-api-endpoint/v1
    api_key: your_api_key
    api_type: openai_compatible  # or raw_completion
```

#### Via UI

Navigate to Settings → Custom Model Settings to configure your custom model.

### Example: Using with Together AI

```bash
CUSTOM_MODEL=togethercomputer/llama-2-70b-chat
CUSTOM_BASE_URL=https://api.together.xyz/v1
CUSTOM_API_KEY=your_together_api_key
CUSTOM_API_TYPE=openai_compatible
```

### Example: Using with Anthropic (Non-standard Endpoint)

```bash
CUSTOM_MODEL=claude-3-opus-20240229
CUSTOM_BASE_URL=https://alternative-claude-api.com
CUSTOM_API_KEY=your_api_key
CUSTOM_API_TYPE=openai_compatible
```

### Example: Using with Local LLM Server

```bash
CUSTOM_MODEL=my-local-model
CUSTOM_BASE_URL=http://localhost:8000
CUSTOM_API_TYPE=raw_completion
```

### Adding Support for More API Types

If you need to support another API format:

1. Extend the `CustomProvider` class in `integrations/llm.py`
2. Add your new API type to the supported types
3. Implement the request/response handling for that API type

```python
elif self.api_type == "your_api_type":
    # Your custom API handling here
    payload = {
        # Your payload format
    }
    
    response = requests.post(
        self.base_url + "/your_endpoint",
        headers=self.headers,
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        return data["your_response_field"]
```

## Best Development Practices

1. **Testing**: Write tests for your components, especially when adding new features
2. **Logging**: Use the built-in logger to add useful logs for debugging
3. **Configuration**: Keep sensitive information in environment variables, not code
4. **Documentation**: Document your modifications, especially APIs and configuration options
5. **Modular Design**: Follow the modular architecture to keep components isolated
