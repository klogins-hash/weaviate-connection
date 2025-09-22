# Optimized Weaviate Vector Database System

This project provides a complete, production-ready vector database system that combines:

- **Weaviate** for vector storage and search
- **Cohere** for high-quality embeddings  
- **Claude (Anthropic)** for intelligent LLM operations

Perfect for building semantic search, RAG (Retrieval-Augmented Generation), and AI-powered applications.

## 🚀 Features

- ✅ **Automatic embedding generation** with Cohere's latest models
- ✅ **Intelligent metadata extraction** using Claude LLM
- ✅ **Optimized semantic search** with query enhancement
- ✅ **Context-aware question answering** (RAG implementation)
- ✅ **Batch document processing** for large datasets
- ✅ **Production-ready collection management**
- ✅ **Comprehensive error handling and logging**
- ✅ **Interactive demo and testing tools**

## 📋 Prerequisites

You'll need API keys for:

- **Weaviate Cloud** (or self-hosted instance)
- **Cohere** (for embeddings) - Get yours at [cohere.ai](https://cohere.ai)
- **Anthropic** (for Claude) - Get yours at [console.anthropic.com](https://console.anthropic.com)

## 🛠️ Setup

### 1. Environment Setup

```bash
# Clone the repository (if from GitHub)
git clone https://github.com/klogins-hash/weaviate-connection.git
cd weaviate-connection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# Weaviate Configuration
WEAVIATE_URL=your_weaviate_cluster_url_here
WEAVIATE_API_KEY=your_api_key_here

# Cohere Configuration (for embeddings)
COHERE_API_KEY=your_cohere_api_key_here

# Anthropic Configuration (for Claude LLM)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🎯 Quick Start

### Test Individual Services

```bash
# Test Weaviate connection
python weaviate_client.py

# Test Cohere embeddings
python embedding_service.py

# Test Claude LLM
python llm_service.py
```

### Run Complete Demo

```bash
# Full workflow demonstration
python demo_workflow.py

# Interactive Q&A mode
python demo_workflow.py interactive
```

### Use the Optimized Service

```python
from optimized_weaviate import OptimizedWeaviateService

# Initialize the service
service = OptimizedWeaviateService()

# Connect to all services
service.connect()

# Create a collection
service.create_collection("my_docs", "My document collection")

# Add a document (automatic embedding + metadata)
doc_id = service.add_document(
    collection_name="my_docs",
    content="Your document content here...",
    title="Document Title",
    source="your_source"
)

# Semantic search
results = service.semantic_search(
    collection_name="my_docs",
    query="What is this about?",
    limit=5
)

# Ask questions (RAG)
answer = service.ask_question(
    collection_name="my_docs",
    question="What are the main points?"
)

print(answer['answer'])
service.close()
```

## 📚 Core Components

### 1. WeaviateClient (`weaviate_client.py`)
Basic Weaviate connection and management.

### 2. CohereEmbeddingService (`embedding_service.py`)
High-quality embeddings with features:

- Multiple Cohere models support
- Batch processing (up to 96 texts)
- Automatic text truncation
- Token counting and optimization

### 3. ClaudeLLMService (`llm_service.py`)
Intelligent text processing with:

- Document summarization
- Query optimization for vector search
- Named entity extraction
- Metadata generation
- Question answering

### 4. OptimizedWeaviateService (`optimized_weaviate.py`)
Complete integration with:

- Automatic collection setup
- Batch document processing
- Advanced semantic search
- RAG-based question answering
- Collection statistics and management

## 🔧 Advanced Usage

### Batch Document Processing

```python
documents = [
    {"content": "Document 1 content...", "title": "Doc 1"},
    {"content": "Document 2 content...", "title": "Doc 2"},
    # ... more documents
]

doc_ids = service.add_documents_batch("my_collection", documents)
```

### Custom Collection Properties

```python
custom_properties = [
    {"name": "author", "data_type": "TEXT"},
    {"name": "publication_date", "data_type": "DATE"},
    {"name": "category", "data_type": "TEXT"}
]

service.create_collection(
    "custom_docs", 
    "Custom document collection",
    properties=custom_properties
)
```

### Advanced Search with Filters

```python
# Search with minimum score threshold
results = service.semantic_search(
    collection_name="my_docs",
    query="machine learning concepts",
    limit=10,
    min_score=0.8
)

# Get detailed results
for result in results:
    print(f"Title: {result['properties']['title']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['properties']['content'][:200]}...")
```

## 📊 Model Specifications

### Cohere Embeddings

- **embed-english-v3.0**: 1024 dimensions, 512 max tokens
- **embed-multilingual-v3.0**: 1024 dimensions, 512 max tokens  
- **embed-english-light-v3.0**: 384 dimensions, 512 max tokens

### Claude Models

- **claude-3-5-sonnet-20241022**: Best performance, 200K context
- **claude-3-5-haiku-20241022**: Fast and cost-effective, 200K context
- **claude-3-opus-20240229**: Most capable, 200K context

## 🔍 Troubleshooting

### Connection Issues

```bash
# Test individual services
python weaviate_client.py    # Test Weaviate
python embedding_service.py  # Test Cohere  
python llm_service.py        # Test Claude
```

### Common Problems

1. **API Key Issues**: Ensure all API keys are set in `.env`
2. **Network Issues**: Check internet connection and firewall
3. **Rate Limits**: Cohere has batch limits (96 texts), Claude has rate limits
4. **Memory Issues**: For large datasets, use smaller batch sizes

### Debug Mode

Set environment variable for detailed logging:

```bash
export WEAVIATE_DEBUG=true
python your_script.py
```

## 🚀 Production Deployment

### Performance Optimization

1. **Batch Size**: Optimize based on your data and rate limits
2. **Collection Configuration**: Tune HNSW parameters for your use case
3. **Caching**: Implement embedding caching for repeated content
4. **Monitoring**: Add logging and metrics collection

### Security Best Practices

1. **Environment Variables**: Never commit API keys to version control
2. **Access Control**: Use Weaviate's built-in authentication
3. **Network Security**: Configure proper firewall rules
4. **API Key Rotation**: Regularly rotate your API keys

## 📈 Cost Optimization

### Cohere Pricing (Embeddings)
- Batch requests for better efficiency
- Use lighter models for less critical applications
- Cache embeddings for repeated content

### Anthropic Pricing (Claude)
- Optimize prompts to reduce token usage
- Use appropriate model sizes for different tasks
- Implement response caching where possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. See the repository for license details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the demo workflow for examples
3. Open an issue on GitHub
4. Check the individual service documentation

---

**Built with ❤️ for the AI community**
