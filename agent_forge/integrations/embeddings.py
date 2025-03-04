from typing import List, Dict, Any, Union, Optional, Callable
import os
from loguru import logger
import time
import numpy as np

class EmbeddingConfig:
    """Configuration for embedding generation."""
    def __init__(
        self,
        batch_size: int = 8,
        device: str = None,
        normalize_embeddings: bool = False
    ):
        self.batch_size = batch_size
        # Use CUDA if available and not explicitly set
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        self.normalize_embeddings = normalize_embeddings

def get_embedding(
    text: Union[str, List[str]], 
    model: str = None,
    config: EmbeddingConfig = None
) -> Union[List[float], List[List[float]]]:
    """
    Generate embedding(s) for the given text using the specified model.
    
    Args:
        text: The text or list of texts to generate embeddings for
        model: The model to use for embedding generation
        config: Configuration for embedding generation
        
    Returns:
        A single embedding vector or list of embedding vectors
    """
    # Use environment variable if model not specified
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Create default config if not provided
    if config is None:
        config = EmbeddingConfig()
    
    # Handle single text vs batch
    is_single_text = isinstance(text, str)
    texts = [text] if is_single_text else text
    
    # Normalize whitespace
    texts = [" ".join(t.split()) for t in texts]
    
    # Select provider based on model name
    if "openai" in model or model.startswith("text-embedding"):
        embeddings = get_openai_embedding(texts, model, config)
    elif "cohere" in model:
        embeddings = get_cohere_embedding(texts, model, config)
    elif "huggingface" in model or "/" in model:
        embeddings = get_huggingface_embedding(texts, model, config)
    else:
        # Default to OpenAI
        logger.warning(f"Unknown embedding model '{model}', falling back to OpenAI")
        embeddings = get_openai_embedding(texts)
    
    # Return single embedding if input was single text
    return embeddings[0] if is_single_text else embeddings

def get_openai_embedding(
    texts: List[str], 
    model: str = "text-embedding-ada-002",
    config: EmbeddingConfig = None
) -> List[List[float]]:
    """Get embeddings using OpenAI's API."""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key required")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Handle rate limits with retries
        max_retries = 3
        retry_delay = 5  # seconds
        
        all_embeddings = []
        batch_size = 8 if config is None else config.batch_size
        
        # Process in batches to handle API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model=model,
                        input=batch
                    )
                    # Extract embeddings in the same order as input
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                        logger.warning(f"OpenAI rate limit hit, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
        
        return all_embeddings
        
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai")
        raise ImportError("OpenAI package required for embeddings")

def get_cohere_embedding(
    texts: List[str], 
    model: str = "embed-english-v3.0",
    config: EmbeddingConfig = None
) -> List[List[float]]:
    """Get embeddings using Cohere's API."""
    try:
        import cohere
        
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.error("Cohere API key not provided. Please set COHERE_API_KEY environment variable.")
            raise ValueError("Cohere API key required")
        
        co = cohere.Client(api_key)
        
        batch_size = 96 if config is None else config.batch_size  # Cohere allows larger batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = co.embed(
                texts=batch,
                model=model,
                input_type="search_query"
            )
            
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
        
    except ImportError:
        logger.error("Cohere package not installed. Install with: pip install cohere")
        raise ImportError("Cohere package required for embeddings")

def get_huggingface_embedding(
    texts: List[str], 
    model: str = "sentence-transformers/all-mpnet-base-v2",
    config: EmbeddingConfig = None
) -> List[List[float]]:
    """
    Get embeddings using Hugging Face models.
    Supports sentence-transformers, transformers, and other embedding models.
    """
    if config is None:
        config = EmbeddingConfig()
    
    # Clean the model name (remove huggingface- prefix if present)
    if model.startswith("huggingface-"):
        model = model[11:]
    
    # Check if it's a sentence-transformer model
    is_sentence_transformer = (
        model.startswith("sentence-transformers/") or
        "/" not in model  # Simple model name, try sentence-transformers
    )
    
    try:
        if is_sentence_transformer:
            return _get_sentence_transformer_embedding(texts, model, config)
        else:
            return _get_transformers_embedding(texts, model, config)
    except ImportError as e:
        logger.error(f"Required package not installed: {str(e)}")
        if "sentence_transformers" in str(e):
            raise ImportError("sentence_transformers package required for Sentence Transformers embeddings. Install with: pip install sentence-transformers")
        else:
            raise ImportError("transformers package required for Hugging Face embeddings. Install with: pip install transformers")

def _get_sentence_transformer_embedding(
    texts: List[str],
    model: str,
    config: EmbeddingConfig
) -> List[List[float]]:
    """Get embeddings using sentence-transformers models."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the model (will download if not cached)
        embedding_model = SentenceTransformer(model, device=config.device)
        
        # Generate embeddings in batches
        embeddings = embedding_model.encode(
            texts, 
            batch_size=config.batch_size,
            convert_to_tensor=False,
            normalize_embeddings=config.normalize_embeddings
        )
        
        return embeddings.tolist()
        
    except ImportError:
        raise ImportError("sentence_transformers")

def _get_transformers_embedding(
    texts: List[str],
    model: str,
    config: EmbeddingConfig
) -> List[List[float]]:
    """Get embeddings using general transformers models."""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model).to(config.device)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i:i + config.batch_size]
            
            # Tokenize
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(config.device)
            
            # Compute embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                
            # Use mean pooling for sentence representation
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize if needed
            if config.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            # Convert to list
            batch_embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
        
    except ImportError:
        raise ImportError("transformers")

# LangChain integration
try:
    from langchain.embeddings.base import Embeddings
    
    class UnifiedEmbeddings(Embeddings):
        """LangChain compatible embedding class that wraps our unified embedding interface."""
        
        def __init__(
            self, 
            model: str = None,
            batch_size: int = 8,
            device: str = None,
            normalize_embeddings: bool = False
        ):
            """Initialize the embedding class with model and configuration."""
            self.model = model
            self.config = EmbeddingConfig(
                batch_size=batch_size,
                device=device,
                normalize_embeddings=normalize_embeddings
            )
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed a list of documents."""
            return get_embedding(texts, self.model, self.config)
            
        def embed_query(self, text: str) -> List[float]:
            """Embed a single query."""
            return get_embedding(text, self.model, self.config)
    
    # Register specific provider classes for direct compatibility
    class UnifiedOpenAIEmbeddings(UnifiedEmbeddings):
        """OpenAI embeddings using our unified interface."""
        def __init__(
            self, 
            model: str = "text-embedding-ada-002",
            **kwargs
        ):
            super().__init__(model=model, **kwargs)
    
    class UnifiedCohereEmbeddings(UnifiedEmbeddings):
        """Cohere embeddings using our unified interface."""
        def __init__(
            self, 
            model: str = "embed-english-v3.0",
            **kwargs
        ):
            super().__init__(model=model, **kwargs)
    
    class UnifiedHuggingFaceEmbeddings(UnifiedEmbeddings):
        """HuggingFace embeddings using our unified interface."""
        def __init__(
            self, 
            model: str = "sentence-transformers/all-mpnet-base-v2",
            **kwargs
        ):
            # Ensure the model has huggingface- prefix for our router
            if not model.startswith("huggingface-") and not any(x in model for x in ["openai", "cohere"]):
                model = f"huggingface-{model}"
            super().__init__(model=model, **kwargs)

except ImportError:
    logger.warning("LangChain not installed. LangChain integrations will not be available.")
    logger.warning("Install with: pip install langchain")
