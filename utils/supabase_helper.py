"""
Helper functions for Supabase integration, particularly for vector search.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("supabase_helper")

class SupabaseVectorStore:
    """
    Class for interacting with Supabase Vector Store using the pgvector extension.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, table_name: str = "embeddings", schema: str = "public"):
        """
        Initialize the Supabase vector store.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            table_name: Name of the table containing embeddings
            schema: Database schema
        """
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self.table_name = table_name
        self.schema = schema
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Supabase vector store for table {self.table_name}")
    
    def create_table_if_not_exists(self, embedding_dimensions: int = 1536):
        """
        Create the vector embeddings table if it doesn't exist.
        
        Args:
            embedding_dimensions: Dimension of the embedding vectors
        
        Returns:
            bool: True if successful
        """
        try:
            # SQL to create pgvector extension and table
            sql = f"""
            -- Enable pgvector extension
            CREATE EXTENSION IF NOT EXISTS vector;
            
            -- Create table if it doesn't exist
            CREATE TABLE IF NOT EXISTS {self.schema}.{self.table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({embedding_dimensions})
            );
            
            -- Create search index
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx ON {self.schema}.{self.table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """
            
            # Execute SQL using Supabase REST API
            response = requests.post(
                f"{self.supabase_url}/rest/v1/rpc/pgexec",
                headers=self.headers,
                json={"query": sql}
            )
            
            if response.status_code == 200:
                logger.info(f"Created vector table {self.table_name} if it didn't exist")
                return True
            else:
                logger.error(f"Failed to create vector table: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating vector table: {e}")
            return False
    
    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add embeddings to the vector store.
        
        Args:
            texts: The text content to store
            embeddings: The embedding vectors
            metadatas: Optional metadata for each text
        
        Returns:
            bool: True if successful
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
            
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("If provided, number of metadata items must match number of texts")
            
        if not metadatas:
            metadatas = [{}] * len(texts)
        
        try:
            # Prepare data for batch insert
            data = []
            for text, embedding, metadata in zip(texts, embeddings, metadatas):
                data.append({
                    "content": text,
                    "embedding": embedding,
                    "metadata": metadata or {}
                })
            
            # Insert using Supabase REST API
            response = requests.post(
                f"{self.supabase_url}/rest/v1/{self.table_name}",
                headers=self.headers,
                json=data
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Added {len(texts)} embeddings to Supabase")
                return True
            else:
                logger.error(f"Failed to add embeddings: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False
    
    def search_embeddings(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
        
        Returns:
            List of documents with similarity scores
        """
        try:
            # Use match_vectors function from pgvector
            sql = f"""
            SELECT 
                id, 
                content,
                metadata,
                1 - (embedding <=> $1) as similarity
            FROM 
                {self.schema}.{self.table_name}
            ORDER BY 
                embedding <=> $1
            LIMIT {limit};
            """
            
            # Execute SQL using Supabase REST API
            response = requests.post(
                f"{self.supabase_url}/rest/v1/rpc/pgexec",
                headers=self.headers,
                json={"query": sql, "params": [query_embedding]}
            )
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Found {len(results)} similar documents")
                return results
            else:
                logger.error(f"Failed to search embeddings: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            embedding_id: The ID of the embedding to delete
        
        Returns:
            bool: True if successful
        """
        try:
            # Delete using Supabase REST API
            response = requests.delete(
                f"{self.supabase_url}/rest/v1/{self.table_name}?id=eq.{embedding_id}",
                headers=self.headers
            )
            
            if response.status_code == 204:
                logger.info(f"Deleted embedding {embedding_id}")
                return True
            else:
                logger.error(f"Failed to delete embedding: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False

def get_supabase_vector_store(config: Optional[Dict[str, Any]] = None) -> Optional[SupabaseVectorStore]:
    """
    Create a Supabase vector store from configuration.
    
    Args:
        config: Configuration dictionary (will use environment variables if not provided)
    
    Returns:
        SupabaseVectorStore instance or None if configuration is missing
    """
    # Get configuration
    supabase_url = None
    supabase_key = None
    table_name = "embeddings"
    schema = "public"
    
    if config:
        supabase_config = config.get("database_config", {}).get("supabase", {})
        supabase_url = supabase_config.get("url")
        supabase_key = supabase_config.get("key")
        table_name = supabase_config.get("table", table_name)
        schema = supabase_config.get("schema", schema)
    
    # Fall back to environment variables
    if not supabase_url:
        supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_key:
        supabase_key = os.environ.get("SUPABASE_KEY")
    if not table_name:
        table_name = os.environ.get("SUPABASE_TABLE", table_name)
    
    # Check if we have the required configuration
    if not (supabase_url and supabase_key):
        logger.warning("Supabase URL and key are required for vector store")
        return None
    
    try:
        # Create the vector store
        vector_store = SupabaseVectorStore(
            supabase_url=supabase_url, 
            supabase_key=supabase_key,
            table_name=table_name,
            schema=schema
        )
        
        # Initialize the vector store
        vector_store.create_table_if_not_exists()
        
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing Supabase vector store: {e}")
        return None
