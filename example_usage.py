#!/usr/bin/env python3
"""
Example usage of the Weaviate client

This script demonstrates how to use the WeaviateClient class
for basic vector database operations.
"""

from weaviate_client import WeaviateClient

def example_operations():
    """Demonstrate basic Weaviate operations."""
    
    # Initialize and connect
    wv_client = WeaviateClient()
    
    if not wv_client.connect():
        print("Failed to connect to Weaviate")
        return
    
    # Test connection
    if not wv_client.test_connection():
        print("Connection test failed")
        return
    
    # Get the raw client for advanced operations
    client = wv_client.get_client()
    
    print("\n🔍 Exploring database capabilities...")
    
    try:
        # Example: Get database statistics
        meta = client.get_meta()
        print(f"🏗️  Database modules: {list(meta.get('modules', {}).keys())}")
        
        # Example: Check cluster status (if available)
        if hasattr(client, 'cluster'):
            nodes = client.cluster.get_nodes_status()
            print(f"📡 Cluster nodes: {len(nodes)}")
        
    except Exception as e:
        print(f"⚠️  Some advanced features not available: {str(e)}")
    
    print("\n✨ Ready for vector operations!")
    print("You can now:")
    print("  - Create schemas")
    print("  - Insert vector data")
    print("  - Perform similarity searches")
    print("  - Run hybrid queries")
    
    # Close connection
    wv_client.close()

if __name__ == "__main__":
    example_operations()
