# Weaviate Database Connection

This project provides a simple Python client to connect to your Weaviate vector database.

## Setup

1. **Create and activate virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables:**
   The credentials are already configured in the `.env` file:
   - `WEAVIATE_URL`: Your Weaviate cluster URL
   - `WEAVIATE_API_KEY`: Your API key for authentication

## Usage

### Basic Connection Test

Run the main client script to test your connection:

```bash
source venv/bin/activate
python weaviate_client.py
```

This will:
- Connect to your Weaviate database
- Test the connection
- Display cluster information
- Show available schemas/classes

### Example Usage

Run the example script to see more operations:

```bash
source venv/bin/activate
python example_usage.py
```

### Using in Your Code

```python
from weaviate_client import WeaviateClient

# Initialize client
wv_client = WeaviateClient()

# Connect
if wv_client.connect():
    # Test connection
    wv_client.test_connection()
    
    # Get raw client for operations
    client = wv_client.get_client()
    
    # Your vector operations here...
    # client.data_object.create(...)
    # client.query.get(...)
    
    # Close when done
    wv_client.close()
```

## Features

- ✅ Secure credential management with environment variables
- ✅ Connection testing and validation
- ✅ Error handling and informative messages
- ✅ Easy-to-use wrapper around the Weaviate client
- ✅ Cluster information and schema inspection

## Next Steps

With the connection established, you can:

1. **Create schemas** for your data
2. **Insert vector embeddings** and objects
3. **Perform similarity searches**
4. **Run hybrid queries** (vector + keyword)
5. **Manage your vector database**

## Troubleshooting

- Ensure your API key and URL are correct
- Check your internet connection
- Verify the Weaviate cluster is running
- Check the console output for specific error messages
