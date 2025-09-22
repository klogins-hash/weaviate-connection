#!/usr/bin/env python3
"""
Demo Workflow

This script demonstrates the complete workflow of the optimized Weaviate service
with Cohere embeddings and Claude LLM integration.
"""

from optimized_weaviate import OptimizedWeaviateService
import time

def demo_workflow():
    """Demonstrate the complete workflow."""
    print("🚀 Starting Optimized Weaviate Demo Workflow")
    print("=" * 60)
    
    # Initialize the service
    print("\n1️⃣ Initializing Services...")
    service = OptimizedWeaviateService()
    
    # Connect to all services
    print("\n2️⃣ Connecting to Services...")
    if not service.connect():
        print("❌ Failed to connect to services")
        return
    
    # Create a collection for demo documents
    collection_name = "knowledge_base"
    print(f"\n3️⃣ Creating Collection: {collection_name}")
    service.create_collection(
        collection_name=collection_name,
        description="Knowledge base for AI and machine learning topics"
    )
    
    # Sample documents to add
    sample_documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": """Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns, make predictions, and improve their performance over time. The three main types of machine learning are supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).""",
            "source": "AI Textbook",
            "category": "fundamentals"
        },
        {
            "title": "Vector Databases Explained",
            "content": """Vector databases are specialized database systems designed to store, index, and query high-dimensional vector data efficiently. They are essential for applications involving embeddings, such as semantic search, recommendation systems, and similarity matching. Vector databases use advanced indexing techniques like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) to enable fast approximate nearest neighbor searches across millions or billions of vectors.""",
            "source": "Database Guide",
            "category": "technology"
        },
        {
            "title": "Natural Language Processing with Transformers",
            "content": """Transformers have revolutionized natural language processing by introducing the attention mechanism, which allows models to focus on relevant parts of input sequences. The transformer architecture, introduced in the 'Attention is All You Need' paper, forms the basis for modern language models like BERT, GPT, and T5. These models can understand context, generate human-like text, and perform various NLP tasks including translation, summarization, and question answering.""",
            "source": "NLP Research",
            "category": "deep_learning"
        },
        {
            "title": "Embeddings and Semantic Search",
            "content": """Embeddings are dense vector representations of data that capture semantic meaning in a continuous vector space. In natural language processing, word embeddings like Word2Vec, GloVe, and more recent contextual embeddings from transformer models represent words or sentences as vectors. These embeddings enable semantic search, where queries and documents are compared based on meaning rather than exact keyword matches, leading to more relevant and intuitive search results.""",
            "source": "ML Engineering",
            "category": "applications"
        },
        {
            "title": "Retrieval-Augmented Generation (RAG)",
            "content": """Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge retrieval. RAG systems first retrieve relevant documents from a knowledge base using vector search, then use this retrieved context to generate more accurate and informative responses. This approach helps overcome the limitations of language models, such as knowledge cutoffs and hallucinations, by grounding responses in factual, up-to-date information.""",
            "source": "AI Research",
            "category": "advanced"
        }
    ]
    
    # Add documents to the collection
    print(f"\n4️⃣ Adding {len(sample_documents)} Documents...")
    doc_ids = service.add_documents_batch(collection_name, sample_documents)
    print(f"✅ Successfully added {len(doc_ids)} documents")
    
    # Wait a moment for indexing
    print("\n⏳ Waiting for indexing to complete...")
    time.sleep(2)
    
    # Demonstrate semantic search
    print("\n5️⃣ Demonstrating Semantic Search")
    print("-" * 40)
    
    search_queries = [
        "How do neural networks learn from data?",
        "What are the benefits of vector search?",
        "Explain attention mechanisms in AI",
        "How does RAG improve language models?"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        results = service.semantic_search(
            collection_name=collection_name,
            query=query,
            limit=3,
            min_score=0.6
        )
        
        for j, result in enumerate(results, 1):
            title = result['properties']['title']
            score = result['score']
            print(f"   {j}. {title} (Score: {score:.3f})")
    
    # Demonstrate question answering
    print("\n6️⃣ Demonstrating Question Answering")
    print("-" * 40)
    
    questions = [
        "What is machine learning and what are its main types?",
        "How do vector databases work and why are they important?",
        "What makes transformers effective for NLP tasks?",
        "How does RAG help improve AI responses?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        qa_result = service.ask_question(
            collection_name=collection_name,
            question=question,
            max_context_docs=3
        )
        
        if 'error' not in qa_result:
            print(f"🤖 Answer: {qa_result['answer'][:300]}...")
            print(f"📊 Confidence: {qa_result['confidence']}")
            print(f"📚 Sources used: {qa_result['context_documents_count']}")
            
            # Show sources
            print("📖 Sources:")
            for source in qa_result['sources']:
                print(f"   - {source['title']} (Score: {source['score']:.3f})")
        else:
            print(f"❌ Error: {qa_result['error']}")
        
        print()  # Add spacing
    
    # Show collection statistics
    print("\n7️⃣ Collection Statistics")
    print("-" * 40)
    stats = service.get_collection_stats(collection_name)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Demonstrate advanced search with filters (if supported)
    print("\n8️⃣ Advanced Features Demo")
    print("-" * 40)
    
    # Search within specific categories
    print("🏷️  Searching within 'technology' category...")
    tech_results = service.semantic_search(
        collection_name=collection_name,
        query="database systems for AI applications",
        limit=5
    )
    
    for result in tech_results:
        if result['properties'].get('category') == 'technology':
            print(f"   ✅ {result['properties']['title']} (Score: {result['score']:.3f})")
    
    # Close connections
    print("\n9️⃣ Cleaning Up...")
    service.close()
    
    print("\n🎉 Demo Workflow Complete!")
    print("=" * 60)
    print("\n💡 Key Features Demonstrated:")
    print("   • Automatic embedding generation with Cohere")
    print("   • Intelligent metadata extraction with Claude")
    print("   • Optimized semantic search")
    print("   • Context-aware question answering")
    print("   • Batch document processing")
    print("   • Collection management and statistics")
    
    print("\n🚀 Your optimized Weaviate system is ready for production use!")

def interactive_demo():
    """Interactive demo for testing queries."""
    print("\n🎯 Interactive Demo Mode")
    print("=" * 40)
    
    service = OptimizedWeaviateService()
    
    if not service.connect():
        print("❌ Failed to connect to services")
        return
    
    collection_name = "knowledge_base"
    
    print(f"Connected to collection: {collection_name}")
    print("Type your questions below (type 'quit' to exit):")
    
    try:
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("🔄 Processing...")
            qa_result = service.ask_question(
                collection_name=collection_name,
                question=question,
                max_context_docs=3
            )
            
            if 'error' not in qa_result:
                print(f"\n🤖 Answer: {qa_result['answer']}")
                print(f"\n📊 Confidence: {qa_result['confidence']}")
                print(f"📚 Sources: {qa_result['context_documents_count']}")
            else:
                print(f"❌ Error: {qa_result['error']}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    finally:
        service.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        demo_workflow()
