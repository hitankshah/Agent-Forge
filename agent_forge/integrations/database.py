from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
from loguru import logger

class BaseVectorStore(ABC):
    """Abstract base class for vector database providers."""
    
    @abstractmethod
    def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store text with embedding and optional metadata, return ID."""
        pass
    
    @abstractmethod
    def query_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar embeddings by vector similarity."""
        pass
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a stored embedding by ID."""
        pass
    
    @abstractmethod
    def update(self, id: str, text: str = None, embedding: List[float] = None, 
              metadata: Dict[str, Any] = None) -> bool:
        """Update a stored embedding by ID."""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a stored embedding by ID."""
        pass


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB store with configuration."""
        self.collection_name = config.get("collection_name", "agent_forge")
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        except ImportError:
            logger.error("ChromaDB package not installed. Install with: pip install chromadb")
            raise ImportError("ChromaDB package required")
    
    def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store text with embedding and optional metadata in ChromaDB."""
        import uuid
        id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
        
        return id
    
    def query_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar embeddings in ChromaDB by vector similarity."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results to standard format
        formatted_results = []
        for i, (id, document, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
            formatted_results.append({
                'id': id,
                'text': document,
                'similarity': 1.0 - distance,  # Convert distance to similarity
                'metadata': metadata
            })
            
        return formatted_results
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a stored embedding by ID from ChromaDB."""
        try:
            result = self.collection.get(ids=[id])
            
            if not result['ids']:
                return None
                
            return {
                'id': result['ids'][0],
                'text': result['documents'][0],
                'metadata': result['metadatas'][0] if 'metadatas' in result else {}
            }
        except Exception as e:
            logger.error(f"Error getting embedding by ID: {str(e)}")
            return None
    
    def update(self, id: str, text: str = None, embedding: List[float] = None, 
              metadata: Dict[str, Any] = None) -> bool:
        """Update a stored embedding by ID in ChromaDB."""
        try:
            # Get current values first
            current = self.get_by_id(id)
            if not current:
                return False
                
            update_dict = {}
            if text is not None:
                update_dict['documents'] = [text]
            if embedding is not None:
                update_dict['embeddings'] = [embedding]
            if metadata is not None:
                update_dict['metadatas'] = [metadata]
                
            if update_dict:
                self.collection.update(ids=[id], **update_dict)
            return True
        except Exception as e:
            logger.error(f"Error updating embedding: {str(e)}")
            return False
    
    def delete(self, id: str) -> bool:
        """Delete a stored embedding by ID from ChromaDB."""
        try:
            self.collection.delete(ids=[id])
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            return False


class SupabaseVectorStore(BaseVectorStore):
    """Supabase vector store implementation using pgvector."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Supabase store with configuration."""
        self.api_url = config.get("url") or os.getenv("SUPABASE_URL")
        self.api_key = config.get("key") or os.getenv("SUPABASE_KEY")
        self.table_name = config.get("table_name", "embeddings")
        
        if not self.api_url or not self.api_key:
            logger.error("Supabase URL and key required. Set in config or environment variables.")
            raise ValueError("Supabase URL and key required")
            
        try:
            from supabase import create_client
            self.client = create_client(self.api_url, self.api_key)
            logger.info(f"Supabase client initialized with table: {self.table_name}")
        except ImportError:
            logger.error("Supabase package not installed. Install with: pip install supabase")
            raise ImportError("Supabase package required")
    
    def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store text with embedding and optional metadata in Supabase."""
        data = {
            "content": text,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        
        response = self.client.table(self.table_name).insert(data).execute()
        
        if 'error' in response:
            logger.error(f"Error storing in Supabase: {response['error']}")
            raise Exception(f"Supabase error: {response['error']}")
            
        return response['data'][0]['id']
    
    def query_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar embeddings in Supabase by vector similarity using pgvector."""
        # Using pgvector's vector similarity search
        # This requires the pgvector extension to be enabled in Supabase
        query = f"""
        SELECT 
            id, 
            content,
            metadata,
            1 - (embedding <=> '{query_embedding}') as similarity
        FROM {self.table_name}
        ORDER BY embedding <=> '{query_embedding}'
        LIMIT {top_k}
        """
        
        response = self.client.rpc('query_embeddings', {
            'query_embedding': query_embedding,
            'match_count': top_k
        }).execute()
        
        if 'error' in response:
            logger.error(f"Error querying Supabase: {response['error']}")
            return []
            
        results = []
        for item in response['data']:
            results.append({
                'id': item['id'],
                'text': item['content'],
                'similarity': item['similarity'],
                'metadata': item['metadata']
            })
            
        return results
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a stored embedding by ID from Supabase."""
        response = self.client.table(self.table_name).select('*').eq('id', id).execute()
        
        if 'error' in response or not response['data']:
            return None
            
        item = response['data'][0]
        return {
            'id': item['id'],
            'text': item['content'],
            'metadata': item['metadata']
        }
    
    def update(self, id: str, text: str = None, embedding: List[float] = None, 
              metadata: Dict[str, Any] = None) -> bool:
        """Update a stored embedding by ID in Supabase."""
        update_data = {}
        
        if text is not None:
            update_data['content'] = text
        if embedding is not None:
            update_data['embedding'] = embedding
        if metadata is not None:
            update_data['metadata'] = metadata
            
        if not update_data:
            return True  # Nothing to update
            
        response = self.client.table(self.table_name).update(update_data).eq('id', id).execute()
        
        if 'error' in response:
            logger.error(f"Error updating in Supabase: {response['error']}")
            return False
            
        return True
    
    def delete(self, id: str) -> bool:
        """Delete a stored embedding by ID from Supabase."""
        response = self.client.table(self.table_name).delete().eq('id', id).execute()
        
        if 'error' in response:
            logger.error(f"Error deleting from Supabase: {response['error']}")
            return False
            
        return True


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Pinecone store with configuration."""
        self.api_key = config.get("api_key") or os.getenv("PINECONE_API_KEY")
        self.environment = config.get("environment") or os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        self.index_name = config.get("index_name", "agent-forge")
        
        if not self.api_key:
            logger.error("Pinecone API key required. Set in config or environment variables.")
            raise ValueError("Pinecone API key required")
            
        try:
            import pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=config.get("dimension", 1536),  # Default for OpenAI embeddings
                    metric=config.get("metric", "cosine")
                )
                
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone initialized with index: {self.index_name}")
        except ImportError:
            logger.error("Pinecone package not installed. Install with: pip install pinecone-client")
            raise ImportError("Pinecone package required")
    
    def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store text with embedding and optional metadata in Pinecone."""
        import uuid
        id = str(uuid.uuid4())
        
        # Include text in metadata
        metadata = metadata or {}
        metadata['text'] = text
        
        self.index.upsert([(id, embedding, metadata)])
        return id
    
    def query_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar embeddings in Pinecone by vector similarity."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        formatted_results = []
        for match in results['matches']:
            text = match['metadata'].pop('text', '')
            formatted_results.append({
                'id': match['id'],
                'text': text,
                'similarity': match['score'],
                'metadata': match['metadata']
            })
            
        return formatted_results
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a stored embedding by ID from Pinecone."""
        try:
            result = self.index.fetch([id])
            
            if id not in result['vectors']:
                return None
                
            vector = result['vectors'][id]
            text = vector['metadata'].pop('text', '')
            
            return {
                'id': id,
                'text': text,
                'metadata': vector['metadata']
            }
        except Exception as e:
            logger.error(f"Error getting embedding from Pinecone: {str(e)}")
            return None
    
    def update(self, id: str, text: str = None, embedding: List[float] = None, 
              metadata: Dict[str, Any] = None) -> bool:
        """Update a stored embedding by ID in Pinecone."""
        try:
            current = self.get_by_id(id)
            if not current:
                return False
                
            # Prepare new metadata
            current_metadata = current.get('metadata', {})
            
            if metadata:
                current_metadata.update(metadata)
                
            if text is not None:
                current_metadata['text'] = text
                
            # If embedding is None, we need to get the current embedding
            if embedding is None:
                # Pinecone doesn't let us retrieve embeddings, so we can't update just metadata
                logger.warning("Pinecone doesn't support updating just metadata without embeddings")
                return False
                
            # Update with new values
            self.index.upsert([(id, embedding, current_metadata)])
            return True
        except Exception as e:
            logger.error(f"Error updating in Pinecone: {str(e)}")
            return False
    
    def delete(self, id: str) -> bool:
        """Delete a stored embedding by ID from Pinecone."""
        try:
            self.index.delete(ids=[id])
            return True
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            return False


class DatabaseManager:
    """Manager class for selecting and initializing database providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the database manager with configuration."""
        self.config = config
        self.db_type = config.get("type", "memory").lower()
        self.store = self._initialize_store()
    
    def _initialize_store(self) -> BaseVectorStore:
        """Initialize the correct database provider based on configuration."""
        if self.db_type == "supabase":
            return SupabaseVectorStore(self.config.get("supabase", {}))
        elif self.db_type == "pinecone":
            return PineconeVectorStore(self.config.get("pinecone", {}))
        elif self.db_type == "chroma":
            return ChromaDBStore(self.config.get("chroma", {}))
        else:
            logger.warning(f"Unknown database type '{self.db_type}', falling back to ChromaDB")
            return ChromaDBStore(self.config.get("chroma", {}))
    
    def store_text_with_embedding(self, text: str, model: str = None, 
                                metadata: Dict[str, Any] = None) -> str:
        """Store text by generating embedding and saving to the database."""
        from agent_forge.integrations.embeddings import get_embedding
        
        # Use environment variable if model not specified
        if model is None:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            
        embedding = get_embedding(text, model)
        
        metadata = metadata or {}
        metadata['model'] = model
        metadata['timestamp'] = time.time()
        
        return self.store.store_embedding(text, embedding, metadata)
    
    def find_similar(self, query: str, top_k: int = 5, 
                    model: str = None) -> List[Dict[str, Any]]:
        """Find similar content by generating query embedding and searching."""
        from agent_forge.integrations.embeddings import get_embedding
        
        # Use environment variable if model not specified
        if model is None:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            
        query_embedding = get_embedding(query, model)
        return self.store.query_similar(query_embedding, top_k)

    def get_status(self) -> Dict[str, Any]:
        """Get current database status information."""
        status = {
            "type": self.db_type,
            "connected": True,
            "timestamp": time.time()
        }
        
        # Add provider-specific status info
        if self.db_type == "supabase":
            try:
                # Add Supabase-specific status info
                status["url"] = self.config.get("supabase", {}).get("url", "").split("@")[-1]  # Safely show domain only
                status["table"] = self.config.get("supabase", {}).get("table_name", "embeddings")
            except Exception:
                pass
        elif self.db_type == "pinecone":
            try:
                # Add Pinecone-specific status info
                status["environment"] = self.config.get("pinecone", {}).get("environment", "")
                status["index"] = self.config.get("pinecone", {}).get("index_name", "")
            except Exception:
                pass
        elif self.db_type == "chroma":
            try:
                # Add ChromaDB-specific status info
                status["collection"] = self.config.get("chroma", {}).get("collection_name", "")
                status["directory"] = self.config.get("chroma", {}).get("persist_directory", "")
            except Exception:
                pass
                
        return status
    
    def clear_database(self) -> bool:
        """Clear all data from the vector database."""
        try:
            if self.db_type == "chroma":
                # ChromaDB clear implementation
                if hasattr(self.store, "collection"):
                    self.store.collection.delete(where={})
            elif self.db_type == "pinecone":
                # Pinecone clear implementation
                if hasattr(self.store, "index"):
                    self.store.index.delete(delete_all=True)
            elif self.db_type == "supabase":
                # Supabase clear implementation
                if hasattr(self.store, "client") and hasattr(self.store, "table_name"):
                    self.store.client.table(self.store.table_name).delete().execute()
            
            logger.info(f"Cleared database of type {self.db_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")
            return False
