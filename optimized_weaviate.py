#!/usr/bin/env python3
"""
Optimized Weaviate Service

This module provides an optimized Weaviate service that integrates:
- Cohere for embeddings
- Claude for LLM operations
- Advanced vector search capabilities
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

from weaviate_client import WeaviateClient
from embedding_service import CohereEmbeddingService
from llm_service import ClaudeLLMService
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

class OptimizedWeaviateService:
    def __init__(
        self,
        embedding_model: str = "embed-english-v3.0",
        llm_model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the optimized Weaviate service.
        
        Args:
            embedding_model: Cohere embedding model
            llm_model: Claude LLM model
        """
        # Initialize services
        self.weaviate_client = WeaviateClient()
        self.embedding_service = CohereEmbeddingService(embedding_model)
        self.llm_service = ClaudeLLMService(llm_model)
        
        self.client = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to all services."""
        try:
            # Connect to Weaviate
            if not self.weaviate_client.connect():
                return False
            
            self.client = self.weaviate_client.get_client()
            self.connected = True
            
            print("✅ All services connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting services: {str(e)}")
            return False
    
    def create_collection(
        self,
        collection_name: str,
        description: str = "Document collection with semantic search",
        properties: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Create an optimized collection for document storage and search.
        
        Args:
            collection_name: Name of the collection
            description: Description of the collection
            properties: Additional properties for the collection
            
        Returns:
            Success status
        """
        if not self.connected:
            print("❌ Not connected to services")
            return False
        
        try:
            # Default properties for document storage
            default_properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="topics", data_type=DataType.TEXT_ARRAY),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="content_type", data_type=DataType.TEXT),
                Property(name="difficulty", data_type=DataType.TEXT),
                Property(name="word_count", data_type=DataType.INT),
            ]
            
            # Add custom properties if provided
            if properties:
                for prop in properties:
                    default_properties.append(
                        Property(
                            name=prop["name"],
                            data_type=getattr(DataType, prop["data_type"])
                        )
                    )
            
            # Create collection with optimized configuration
            collection = self.client.collections.create(
                name=collection_name,
                description=description,
                properties=default_properties,
                # Configure for optimal performance
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE,
                    dynamic_ef_factor=8,
                    dynamic_ef_max=500,
                    ef_construction=128,
                    max_connections=64
                ),
                # Configure replication for production use
                replication_config=wvc.config.Configure.replication(factor=1)
            )
            
            print(f"✅ Created collection '{collection_name}' with {len(default_properties)} properties")
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection: {str(e)}")
            return False
    
    def add_document(
        self,
        collection_name: str,
        content: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a document to the collection with automatic embedding and metadata generation.
        
        Args:
            collection_name: Target collection
            content: Document content
            title: Document title (auto-generated if None)
            source: Document source
            metadata: Additional metadata
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.connected:
            print("❌ Not connected to services")
            return None
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Generate embedding
            print("🔄 Generating embedding...")
            embedding = self.embedding_service.generate_embedding(content)
            
            # Generate metadata using Claude
            print("🔄 Generating metadata...")
            generated_metadata = self.llm_service.generate_metadata(content)
            
            # Prepare document data
            doc_data = {
                "title": title or generated_metadata.get("title", "Untitled Document"),
                "content": content,
                "source": source or "unknown",
                "created_at": datetime.now(),
                "topics": generated_metadata.get("topics", []),
                "keywords": generated_metadata.get("keywords", []),
                "content_type": generated_metadata.get("content_type", "text"),
                "difficulty": generated_metadata.get("difficulty", "intermediate"),
                "word_count": len(content.split())
            }
            
            # Add custom metadata
            if metadata:
                doc_data.update(metadata)
            
            # Insert document
            doc_id = collection.data.insert(
                properties=doc_data,
                vector=embedding.tolist()
            )
            
            print(f"✅ Added document with ID: {doc_id}")
            return str(doc_id)
            
        except Exception as e:
            print(f"❌ Error adding document: {str(e)}")
            return None
    
    def add_documents_batch(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> List[str]:
        """
        Add multiple documents in batches.
        
        Args:
            collection_name: Target collection
            documents: List of documents with 'content' and optional metadata
            batch_size: Batch size for processing
            
        Returns:
            List of document IDs
        """
        if not self.connected:
            print("❌ Not connected to services")
            return []
        
        try:
            collection = self.client.collections.get(collection_name)
            all_doc_ids = []
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                print(f"🔄 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                # Extract content for batch embedding
                contents = [doc.get('content', '') for doc in batch_docs]
                
                # Generate embeddings in batch
                print("🔄 Generating embeddings...")
                embeddings = self.embedding_service.generate_embeddings_batch(contents)
                
                # Prepare batch data
                batch_data = []
                for doc, embedding in zip(batch_docs, embeddings):
                    content = doc.get('content', '')
                    
                    # Generate metadata for each document
                    generated_metadata = self.llm_service.generate_metadata(content)
                    
                    doc_data = {
                        "title": doc.get('title') or generated_metadata.get("title", "Untitled Document"),
                        "content": content,
                        "source": doc.get('source', 'unknown'),
                        "created_at": datetime.now(),
                        "topics": generated_metadata.get("topics", []),
                        "keywords": generated_metadata.get("keywords", []),
                        "content_type": generated_metadata.get("content_type", "text"),
                        "difficulty": generated_metadata.get("difficulty", "intermediate"),
                        "word_count": len(content.split())
                    }
                    
                    # Add custom metadata
                    for key, value in doc.items():
                        if key not in ['content', 'title', 'source']:
                            doc_data[key] = value
                    
                    batch_data.append({
                        "properties": doc_data,
                        "vector": embedding.tolist()
                    })
                
                # Insert batch
                with collection.batch.dynamic() as batch:
                    for data in batch_data:
                        doc_id = batch.add_object(
                            properties=data["properties"],
                            vector=data["vector"]
                        )
                        all_doc_ids.append(str(doc_id))
                
                print(f"✅ Processed batch {i//batch_size + 1}")
            
            print(f"✅ Added {len(all_doc_ids)} documents total")
            return all_doc_ids
            
        except Exception as e:
            print(f"❌ Error adding documents batch: {str(e)}")
            return []
    
    def semantic_search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using optimized query.
        
        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum results
            min_score: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        if not self.connected:
            print("❌ Not connected to services")
            return []
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Optimize query using Claude
            print("🔄 Optimizing search query...")
            optimized_query = self.llm_service.generate_search_query(query)
            print(f"📝 Optimized query: {optimized_query}")
            
            # Generate query embedding
            print("🔄 Generating query embedding...")
            query_embedding = self.embedding_service.similarity_search_embedding(optimized_query)
            
            # Perform vector search
            results = collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(score=True, distance=True)
            )
            
            # Format results
            formatted_results = []
            for obj in results.objects:
                score = obj.metadata.score if obj.metadata.score else 0.0
                
                if score >= min_score:
                    result = {
                        "id": str(obj.uuid),
                        "score": score,
                        "distance": obj.metadata.distance,
                        "properties": obj.properties
                    }
                    formatted_results.append(result)
            
            print(f"✅ Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error performing semantic search: {str(e)}")
            return []
    
    def ask_question(
        self,
        collection_name: str,
        question: str,
        max_context_docs: int = 5
    ) -> Dict[str, Any]:
        """
        Ask a question and get an AI-generated answer based on relevant documents.
        
        Args:
            collection_name: Collection to search
            question: User question
            max_context_docs: Maximum documents to use as context
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.connected:
            print("❌ Not connected to services")
            return {"error": "Not connected to services"}
        
        try:
            # Search for relevant documents
            print("🔍 Searching for relevant documents...")
            search_results = self.semantic_search(
                collection_name=collection_name,
                query=question,
                limit=max_context_docs,
                min_score=0.7
            )
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # Prepare context documents
            context_docs = []
            for result in search_results:
                props = result["properties"]
                context_docs.append({
                    "title": props.get("title", "Untitled"),
                    "content": props.get("content", ""),
                    "source": props.get("source", "unknown"),
                    "score": result["score"]
                })
            
            # Generate answer using Claude
            print("🤖 Generating answer...")
            answer = self.llm_service.answer_question(question, context_docs)
            
            # Calculate confidence based on search scores
            avg_score = sum(doc["score"] for doc in context_docs) / len(context_docs)
            confidence = "high" if avg_score > 0.8 else "medium" if avg_score > 0.7 else "low"
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "title": doc["title"],
                        "source": doc["source"],
                        "score": doc["score"]
                    }
                    for doc in context_docs
                ],
                "confidence": confidence,
                "context_documents_count": len(context_docs)
            }
            
        except Exception as e:
            print(f"❌ Error answering question: {str(e)}")
            return {"error": str(e)}
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        if not self.connected:
            return {"error": "Not connected"}
        
        try:
            collection = self.client.collections.get(collection_name)
            
            # Get collection info
            response = collection.aggregate.over_all(total_count=True)
            
            return {
                "name": collection_name,
                "total_documents": response.total_count,
                "embedding_dimensions": self.embedding_service.get_embedding_dimensions(),
                "embedding_model": self.embedding_service.model,
                "llm_model": self.llm_service.model
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close all connections."""
        if self.weaviate_client:
            self.weaviate_client.close()
        self.connected = False
        print("🔌 All connections closed")

def main():
    """Test the optimized Weaviate service."""
    print("🚀 Testing Optimized Weaviate Service...")
    
    try:
        # Initialize service
        service = OptimizedWeaviateService()
        
        # Connect
        if not service.connect():
            print("❌ Failed to connect")
            return
        
        # Test collection creation
        collection_name = "test_documents"
        print(f"\n📁 Creating collection '{collection_name}'...")
        service.create_collection(collection_name, "Test collection for documents")
        
        # Test document addition
        print(f"\n📄 Adding test document...")
        doc_id = service.add_document(
            collection_name=collection_name,
            content="Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They enable semantic search, recommendation systems, and similarity matching across various data types.",
            title="Introduction to Vector Databases",
            source="test_system"
        )
        
        if doc_id:
            # Test search
            print(f"\n🔍 Testing semantic search...")
            results = service.semantic_search(
                collection_name=collection_name,
                query="What are vector databases used for?",
                limit=5
            )
            
            for i, result in enumerate(results, 1):
                print(f"Result {i}: {result['properties']['title']} (Score: {result['score']:.3f})")
            
            # Test question answering
            print(f"\n❓ Testing question answering...")
            qa_result = service.ask_question(
                collection_name=collection_name,
                question="What are the main use cases for vector databases?"
            )
            
            print(f"Answer: {qa_result['answer'][:200]}...")
            print(f"Confidence: {qa_result['confidence']}")
            print(f"Sources: {len(qa_result['sources'])}")
        
        # Get collection stats
        print(f"\n📊 Collection statistics...")
        stats = service.get_collection_stats(collection_name)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n✨ Optimized Weaviate service is ready!")
        
        # Close connections
        service.close()
        
    except Exception as e:
        print(f"❌ Error testing service: {str(e)}")
        print("💡 Make sure all API keys are set in the .env file")

if __name__ == "__main__":
    main()
