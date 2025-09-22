#!/usr/bin/env python3
"""
Weaviate Database Connection Client

This script establishes a connection to a Weaviate vector database
and provides basic functionality for testing the connection.
"""

import os
import weaviate
from dotenv import load_dotenv
import weaviate.classes as wvc

class WeaviateClient:
    def __init__(self):
        """Initialize the Weaviate client with credentials from environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        
        self.url = os.getenv('WEAVIATE_URL')
        self.api_key = os.getenv('WEAVIATE_API_KEY')
        
        if not self.url or not self.api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in environment variables")
        
        self.client = None
        
    def connect(self):
        """Establish connection to Weaviate database."""
        try:
            # Initialize Weaviate client with v4 API
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=wvc.init.Auth.api_key(self.api_key)
            )
            
            print(f"✅ Successfully connected to Weaviate at: {self.url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Weaviate: {str(e)}")
            return False
    
    def test_connection(self):
        """Test the connection by checking if the database is ready."""
        if not self.client:
            print("❌ No active connection. Please call connect() first.")
            return False
            
        try:
            # Check if Weaviate is ready
            ready = self.client.is_ready()
            if ready:
                print("✅ Weaviate database is ready and accessible")
                
                # Get cluster metadata
                meta = self.client.get_meta()
                print(f"📊 Weaviate version: {meta.get('version', 'Unknown')}")
                
                # List existing collections (classes in v4)
                collections = list(self.client.collections.list_all().keys())
                print(f"📋 Number of collections in database: {len(collections)}")
                
                if collections:
                    print("📝 Available collections:")
                    for collection_name in collections:
                        print(f"   - {collection_name}")
                else:
                    print("📝 No collections found in database")
                
                return True
            else:
                print("❌ Weaviate database is not ready")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
    
    def get_client(self):
        """Return the Weaviate client instance."""
        return self.client
    
    def close(self):
        """Close the connection to Weaviate."""
        if self.client:
            # Close the connection properly in v4
            self.client.close()
            self.client = None
            print("🔌 Connection closed")

def main():
    """Main function to demonstrate Weaviate connection."""
    print("🚀 Initializing Weaviate connection...")
    
    # Create client instance
    wv_client = WeaviateClient()
    
    # Connect to database
    if wv_client.connect():
        # Test the connection
        wv_client.test_connection()
        
        # Keep the client available for further operations
        print("\n💡 Connection established successfully!")
        print("You can now use the client for vector operations.")
        
        # Example: You can access the client like this:
        # client = wv_client.get_client()
        # Use client for your vector database operations
        
        # Close the connection properly
        wv_client.close()
        
    else:
        print("❌ Failed to establish connection")
        return False
    
    return True

if __name__ == "__main__":
    main()
