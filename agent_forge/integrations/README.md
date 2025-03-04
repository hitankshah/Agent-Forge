# Embeddings Integration

This module provides a unified interface for generating text embeddings from multiple providers with advanced features and LangChain compatibility.

## Capabilities

The embeddings module can:

1. **Generate text embeddings** from three different providers:
   - OpenAI (default model: `text-embedding-ada-002`)
   - Cohere (default model: `embed-english-v3.0`) 
   - Hugging Face with comprehensive model support:
     - Sentence Transformers (default: `sentence-transformers/all-mpnet-base-v2`)
     - General Transformers models with mean pooling
     - Custom model architecture support

2. **Auto-select the provider** based on the model name
   - Models containing "openai" or starting with "text-embedding" use OpenAI
   - Models containing "cohere" use Cohere
   - Models containing "huggingface" or containing "/" use Hugging Face
   - Falls back to OpenAI for unknown models

3. **Advanced features**:
   - Batch processing for efficient embedding generation
   - GPU acceleration with configurable device selection
   - Embedding normalization option
   - Automatic handling of rate limits with retries and backoff

4. **LangChain integration**:
   - Fully compatible with LangChain's Embeddings interface
   - Drop-in replacement for LangChain embedding providers
   - Specialized classes for each provider

## Basic Usage

```python
from agent_forge.integrations.embeddings import get_embedding

# Single text embedding (OpenAI by default)
embedding = get_embedding("Your text here")

# Batch processing
embeddings = get_embedding(["Text 1", "Text 2", "Text 3"])

# Specify different models
openai_embedding = get_embedding("Your text here", model="text-embedding-ada-002")
cohere_embedding = get_embedding("Your text here", model="cohere-embed-english-v3.0")
hf_embedding = get_embedding("Your text here", model="huggingface-sentence-transformers/all-mpnet-base-v2")
```

## Advanced Usage

```python
from agent_forge.integrations.embeddings import get_embedding, EmbeddingConfig

# Configure embedding generation
config = EmbeddingConfig(
    batch_size=16,           # Process 16 texts at once
    device="cuda:0",         # Use first GPU (or "cpu" for CPU)
    normalize_embeddings=True # L2 normalize embeddings
)

# Get embeddings with configuration
embeddings = get_embedding(
    ["Text 1", "Text 2", "Text 3"],
    model="huggingface-sentence-transformers/all-MiniLM-L6-v2",
    config=config
)
```

## LangChain Integration

```python
from agent_forge.integrations.embeddings import UnifiedEmbeddings, UnifiedHuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Generic embeddings
embeddings = UnifiedEmbeddings(model="text-embedding-ada-002")

# Provider-specific embeddings
hf_embeddings = UnifiedHuggingFaceEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

# Use with LangChain components
db = Chroma.from_texts(
    ["Text 1", "Text 2", "Text 3"],
    embeddings
)
results = db.similarity_search("query text", k=2)
```

## Setup

Set the required environment variables for the providers you plan to use:

```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
EMBEDDING_MODEL=your_preferred_default_model
```

## Requirements

Core dependencies:
- `loguru`
- `numpy`

Provider-specific dependencies:
- OpenAI: `pip install openai`
- Cohere: `pip install cohere`
- Hugging Face (basic): `pip install sentence-transformers`
- Hugging Face (advanced): `pip install transformers torch`
- LangChain integration: `pip install langchain`














