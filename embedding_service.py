#!/usr/bin/env python3
"""
Cohere Embedding Service

This module provides a service for generating embeddings using Cohere's API.
Optimized for use with Weaviate vector database.
"""

import os
import cohere
import numpy as np
import tiktoken
from typing import List, Union, Dict, Any, Optional
from dotenv import load_dotenv

class CohereEmbeddingService:
    def __init__(self, model: str = "embed-english-v3.0"):
        """
        Initialize the Cohere embedding service.
        
        Args:
            model: Cohere embedding model to use
                   Options: "embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0"
        """
        load_dotenv()
        
        self.api_key = os.getenv('COHERE_API_KEY')
        if not self.api_key or self.api_key == 'your_cohere_api_key_here':
            raise ValueError("COHERE_API_KEY must be set in environment variables")
        
        self.model = model
        self.client = cohere.Client(self.api_key)
        
        # Model specifications
        self.model_specs = {
            "embed-english-v3.0": {"dimensions": 1024, "max_tokens": 512},
            "embed-multilingual-v3.0": {"dimensions": 1024, "max_tokens": 512},
            "embed-english-light-v3.0": {"dimensions": 384, "max_tokens": 512}
        }
        
        # Initialize tokenizer for text length estimation
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            print("⚠️  Warning: Could not load tokenizer. Token counting will be approximate.")
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the current model."""
        return self.model_specs.get(self.model, {}).get("dimensions", 1024)
    
    def get_max_tokens(self) -> int:
        """Get the maximum token limit for the current model."""
        return self.model_specs.get(self.model, {}).get("max_tokens", 512)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate if tokenizer not available).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens (uses model default if None)
            
        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.get_max_tokens()
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        else:
            # Rough approximation
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
    
    def generate_embedding(self, text: str, input_type: str = "search_document") -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            input_type: Type of input for optimization
                       Options: "search_document", "search_query", "classification", "clustering"
        
        Returns:
            Embedding vector as numpy array
        """
        # Truncate text if necessary
        text = self.truncate_text(text)
        
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=input_type,
                embedding_types=["float"]
            )
            
            embedding = np.array(response.embeddings.float[0])
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        input_type: str = "search_document",
        batch_size: int = 96
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            input_type: Type of input for optimization
            batch_size: Number of texts to process per batch (Cohere limit is 96)
        
        Returns:
            List of embedding vectors as numpy arrays
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Truncate texts in batch
            batch_texts = [self.truncate_text(text) for text in batch_texts]
            
            try:
                response = self.client.embed(
                    texts=batch_texts,
                    model=self.model,
                    input_type=input_type,
                    embedding_types=["float"]
                )
                
                batch_embeddings = [np.array(emb) for emb in response.embeddings.float]
                all_embeddings.extend(batch_embeddings)
                
                print(f"✅ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                raise
        
        return all_embeddings
    
    def similarity_search_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding optimized for similarity search queries.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.generate_embedding(query, input_type="search_query")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "dimensions": self.get_embedding_dimensions(),
            "max_tokens": self.get_max_tokens(),
            "provider": "Cohere"
        }

def main():
    """Test the embedding service."""
    print("🚀 Testing Cohere Embedding Service...")
    
    try:
        # Initialize service
        embedding_service = CohereEmbeddingService()
        
        # Display model info
        info = embedding_service.get_model_info()
        print(f"📊 Model: {info['model']}")
        print(f"📏 Dimensions: {info['dimensions']}")
        print(f"🔢 Max tokens: {info['max_tokens']}")
        
        # Test single embedding
        test_text = "This is a test document for embedding generation."
        print(f"\n🧪 Testing single embedding...")
        print(f"Input: {test_text}")
        
        embedding = embedding_service.generate_embedding(test_text)
        print(f"✅ Generated embedding with shape: {embedding.shape}")
        print(f"📈 First 5 values: {embedding[:5]}")
        
        # Test batch embeddings
        test_texts = [
            "Machine learning is transforming technology.",
            "Vector databases enable semantic search.",
            "Embeddings capture semantic meaning in text."
        ]
        
        print(f"\n🧪 Testing batch embeddings...")
        batch_embeddings = embedding_service.generate_embeddings_batch(test_texts)
        print(f"✅ Generated {len(batch_embeddings)} embeddings")
        
        # Test similarity search embedding
        query = "What is machine learning?"
        query_embedding = embedding_service.similarity_search_embedding(query)
        print(f"\n🔍 Query embedding shape: {query_embedding.shape}")
        
        print("\n✨ Cohere embedding service is ready!")
        
    except Exception as e:
        print(f"❌ Error testing embedding service: {str(e)}")
        print("💡 Make sure to set your COHERE_API_KEY in the .env file")

if __name__ == "__main__":
    main()
