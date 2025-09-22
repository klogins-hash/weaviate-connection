#!/usr/bin/env python3
"""
Claude LLM Service

This module provides a service for interacting with Claude (Anthropic) LLM.
Optimized for use with Weaviate vector database operations.
"""

import os
import anthropic
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

class ClaudeLLMService:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Claude LLM service.
        
        Args:
            model: Claude model to use
                   Options: "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"
        """
        load_dotenv()
        
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key or self.api_key == 'your_anthropic_api_key_here':
            raise ValueError("ANTHROPIC_API_KEY must be set in environment variables")
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Model specifications
        self.model_specs = {
            "claude-3-5-sonnet-20241022": {"max_tokens": 200000, "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015},
            "claude-3-5-haiku-20241022": {"max_tokens": 200000, "cost_per_1k_input": 0.00025, "cost_per_1k_output": 0.00125},
            "claude-3-opus-20240229": {"max_tokens": 200000, "cost_per_1k_input": 0.015, "cost_per_1k_output": 0.075}
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        specs = self.model_specs.get(self.model, {})
        return {
            "model": self.model,
            "max_tokens": specs.get("max_tokens", 200000),
            "cost_per_1k_input": specs.get("cost_per_1k_input", 0.003),
            "cost_per_1k_output": specs.get("cost_per_1k_output", 0.015),
            "provider": "Anthropic"
        }
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0
    ) -> str:
        """
        Generate a response using Claude.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            
        Returns:
            Generated response text
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            raise
    
    def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Summarize a list of documents, optionally focused on a query.
        
        Args:
            documents: List of documents with 'content' and optional metadata
            query: Optional query to focus the summary
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Summary text
        """
        # Prepare document content
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', str(doc))
            title = doc.get('title', f'Document {i}')
            doc_texts.append(f"## {title}\n{content}")
        
        combined_docs = "\n\n".join(doc_texts)
        
        if query:
            prompt = f"""Please provide a comprehensive summary of the following documents, focusing specifically on information relevant to this query: "{query}"

Documents:
{combined_docs}

Summary:"""
            system_prompt = "You are an expert at analyzing and summarizing documents. Focus on the most relevant information related to the user's query while maintaining accuracy and completeness."
        else:
            prompt = f"""Please provide a comprehensive summary of the following documents:

Documents:
{combined_docs}

Summary:"""
            system_prompt = "You are an expert at analyzing and summarizing documents. Provide clear, concise, and comprehensive summaries that capture the key points and insights."
        
        return self.generate_response(prompt, system_prompt, max_tokens)
    
    def generate_search_query(
        self,
        user_question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate an optimized search query for vector search.
        
        Args:
            user_question: Original user question
            context: Optional context about the search domain
            
        Returns:
            Optimized search query
        """
        context_text = f"\nContext: {context}" if context else ""
        
        prompt = f"""Given the following user question, generate an optimized search query that would be most effective for finding relevant documents in a vector database.

User Question: {user_question}{context_text}

The search query should:
- Use keywords and phrases that are likely to appear in relevant documents
- Be concise but comprehensive
- Focus on the core concepts and entities
- Avoid question words and instead use declarative terms

Optimized Search Query:"""
        
        system_prompt = "You are an expert at creating effective search queries for vector databases. Generate queries that maximize semantic similarity matching."
        
        return self.generate_response(prompt, system_prompt, max_tokens=200).strip()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        prompt = f"""Extract named entities from the following text and categorize them. Return the results in a structured format.

Text: {text}

Please identify and categorize entities such as:
- PERSON: People's names
- ORGANIZATION: Companies, institutions, organizations
- LOCATION: Places, cities, countries
- DATE: Dates and time expressions
- PRODUCT: Products, services, technologies
- CONCEPT: Important concepts, topics, themes

Format your response as a JSON-like structure with entity types as keys and lists of entities as values."""
        
        system_prompt = "You are an expert at named entity recognition. Extract entities accurately and categorize them appropriately."
        
        response = self.generate_response(prompt, system_prompt, max_tokens=1000)
        
        # Parse the response (simplified - in production you might want more robust parsing)
        try:
            import json
            # Try to extract JSON from the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback: return the raw response
        return {"raw_response": [response]}
    
    def generate_metadata(self, content: str) -> Dict[str, Any]:
        """
        Generate metadata for content to enhance vector search.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary of metadata
        """
        prompt = f"""Analyze the following content and generate metadata that would be useful for search and categorization:

Content: {content}

Please provide:
1. A concise title (max 10 words)
2. 3-5 key topics/themes
3. 3-5 important keywords
4. Content type (e.g., article, documentation, tutorial, etc.)
5. Difficulty level (beginner, intermediate, advanced)
6. A brief description (1-2 sentences)

Format as JSON."""
        
        system_prompt = "You are an expert at content analysis and metadata generation. Create metadata that enhances searchability and organization."
        
        response = self.generate_response(prompt, system_prompt, max_tokens=500)
        
        # Parse JSON response
        try:
            import json
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback metadata
        return {
            "title": "Generated Content",
            "topics": ["general"],
            "keywords": ["content"],
            "content_type": "text",
            "difficulty": "intermediate",
            "description": "Content processed by Claude LLM service"
        }
    
    def answer_question(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> str:
        """
        Answer a question based on provided context documents.
        
        Args:
            question: User question
            context_documents: Relevant documents for context
            max_tokens: Maximum tokens for the answer
            
        Returns:
            Answer text
        """
        # Prepare context
        context_texts = []
        for i, doc in enumerate(context_documents, 1):
            content = doc.get('content', str(doc))
            title = doc.get('title', f'Document {i}')
            context_texts.append(f"[{i}] {title}: {content}")
        
        context = "\n\n".join(context_texts)
        
        prompt = f"""Based on the following context documents, please answer the user's question. If the answer cannot be found in the provided context, please say so clearly.

Context Documents:
{context}

Question: {question}

Answer:"""
        
        system_prompt = "You are a helpful assistant that answers questions based on provided context. Be accurate, cite sources when possible, and acknowledge when information is not available in the context."
        
        return self.generate_response(prompt, system_prompt, max_tokens)

def main():
    """Test the LLM service."""
    print("🚀 Testing Claude LLM Service...")
    
    try:
        # Initialize service
        llm_service = ClaudeLLMService()
        
        # Display model info
        info = llm_service.get_model_info()
        print(f"📊 Model: {info['model']}")
        print(f"🔢 Max tokens: {info['max_tokens']}")
        print(f"💰 Cost per 1K input tokens: ${info['cost_per_1k_input']}")
        print(f"💰 Cost per 1K output tokens: ${info['cost_per_1k_output']}")
        
        # Test basic response generation
        print(f"\n🧪 Testing basic response generation...")
        response = llm_service.generate_response(
            "What are the key benefits of vector databases?",
            max_tokens=500
        )
        print(f"✅ Response: {response[:200]}...")
        
        # Test search query generation
        print(f"\n🧪 Testing search query generation...")
        query = llm_service.generate_search_query(
            "How do I optimize my machine learning model?",
            "Machine learning and AI development"
        )
        print(f"✅ Optimized query: {query}")
        
        # Test metadata generation
        print(f"\n🧪 Testing metadata generation...")
        sample_content = "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They enable semantic search and similarity matching."
        metadata = llm_service.generate_metadata(sample_content)
        print(f"✅ Generated metadata: {metadata}")
        
        print("\n✨ Claude LLM service is ready!")
        
    except Exception as e:
        print(f"❌ Error testing LLM service: {str(e)}")
        print("💡 Make sure to set your ANTHROPIC_API_KEY in the .env file")

if __name__ == "__main__":
    main()
