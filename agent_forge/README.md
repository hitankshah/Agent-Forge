# Agent Forge

An AI agent builder platform inspired by Archon, allowing you to create, customize, and deploy autonomous AI agents.

## Features

- Design and create AI agents with specific capabilities
- Agent self-improvement through feedback loops
- Interactive UI for agent management
- **Support for ANY LLM model** - bring your own models or use our built-in providers
- Multiple built-in LLM providers (OpenAI, Anthropic, Google, Cohere, Ollama)
- Vector database integration (Supabase, Pinecone, ChromaDB)
- Docker containerization for easy deployment

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and add your API keys:
   ```
   cp .env.template .env
   ```
   Then edit the `.env` file with your actual API keys

4. Run the application:
   ```
   python -m agent_forge.main
   ```

### Using Docker

Start the complete environment with Docker Compose:

```bash
docker-compose up -d
```

This will start:
- The main Agent Forge application
- ChromaDB vector database
- Ollama local LLM service (optional, with GPU support)

Access the UI at: http://localhost:8501

## Available LLM Providers

- **OpenAI**: GPT-4, GPT-3.5 models
- **Anthropic**: Claude models
- **Google**: Gemini models
- **Cohere**: Command models
- **Ollama**: Local open-source models (llama2, mistral, etc.)
- **Custom**: Connect to ANY model with a compatible API

## Using Custom Models

Agent Forge supports using any LLM model that has an API endpoint:

1. Configure the model in your `.env` file:
   ```
   CUSTOM_MODEL=your_model_name
   CUSTOM_BASE_URL=https://your-api-endpoint/v1
   CUSTOM_API_KEY=your_api_key
   CUSTOM_API_TYPE=openai_compatible
   ```

2. Or configure directly in the UI under Settings → Custom Model Settings

3. Supported API types:
   - `openai_compatible`: APIs following the OpenAI format (most hosted models)
   - `raw_completion`: Simple request/response APIs that accept a prompt field

## Vector Database Options

- **ChromaDB**: Local vector database (default)
- **Supabase**: PostgreSQL with pgvector extension
- **Pinecone**: Managed vector database service

## Development

See the [development guide](docs/development.md) for detailed information about the architecture, components, and how to extend the system.

## License

MIT License
