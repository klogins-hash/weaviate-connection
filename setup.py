#!/usr/bin/env python3
"""
Setup Script for Optimized Weaviate System

This script helps users set up their environment and test their API keys.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print setup header."""
    print("🚀 Optimized Weaviate Setup")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("🔄 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("🔄 Installing dependencies...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = Path("venv/Scripts/pip")
    else:  # Unix/Linux/macOS
        pip_path = Path("venv/bin/pip")
    
    try:
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_env_file():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("🔄 Creating .env file from template...")
            env_file.write_text(env_example.read_text())
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your actual API keys")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True

def test_api_keys():
    """Test if API keys are configured."""
    print("\n🔍 Checking API Key Configuration")
    print("-" * 40)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    keys_to_check = [
        ("WEAVIATE_URL", "Weaviate cluster URL"),
        ("WEAVIATE_API_KEY", "Weaviate API key"),
        ("COHERE_API_KEY", "Cohere API key"),
        ("ANTHROPIC_API_KEY", "Anthropic API key")
    ]
    
    all_configured = True
    
    for key, description in keys_to_check:
        value = os.getenv(key)
        if not value or value.startswith('your_') or value == 'your_api_key_here':
            print(f"❌ {description}: Not configured")
            all_configured = False
        else:
            # Show partial key for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {description}: {masked_value}")
    
    return all_configured

def test_services():
    """Test individual services."""
    print("\n🧪 Testing Services")
    print("-" * 40)
    
    # Test Weaviate connection
    print("Testing Weaviate connection...")
    try:
        from weaviate_client import WeaviateClient
        client = WeaviateClient()
        if client.connect():
            client.test_connection()
            client.close()
            print("✅ Weaviate: Connected successfully")
        else:
            print("❌ Weaviate: Connection failed")
            return False
    except Exception as e:
        print(f"❌ Weaviate: Error - {str(e)}")
        return False
    
    # Test Cohere (if API key is set)
    cohere_key = os.getenv('COHERE_API_KEY')
    if cohere_key and not cohere_key.startswith('your_'):
        print("Testing Cohere embeddings...")
        try:
            from embedding_service import CohereEmbeddingService
            embedding_service = CohereEmbeddingService()
            test_embedding = embedding_service.generate_embedding("test text")
            print(f"✅ Cohere: Generated embedding with {len(test_embedding)} dimensions")
        except Exception as e:
            print(f"❌ Cohere: Error - {str(e)}")
            return False
    else:
        print("⚠️  Cohere: API key not configured, skipping test")
    
    # Test Claude (if API key is set)
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    if claude_key and not claude_key.startswith('your_'):
        print("Testing Claude LLM...")
        try:
            from llm_service import ClaudeLLMService
            llm_service = ClaudeLLMService()
            response = llm_service.generate_response("Hello, this is a test.", max_tokens=50)
            print(f"✅ Claude: Generated response ({len(response)} characters)")
        except Exception as e:
            print(f"❌ Claude: Error - {str(e)}")
            return False
    else:
        print("⚠️  Claude: API key not configured, skipping test")
    
    return True

def main():
    """Main setup function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Set up virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Set up .env file
    if not setup_env_file():
        return False
    
    # Test API keys
    keys_configured = test_api_keys()
    
    if keys_configured:
        # Test services
        if test_services():
            print("\n🎉 Setup Complete!")
            print("=" * 50)
            print("Your optimized Weaviate system is ready to use!")
            print("\nNext steps:")
            print("1. Run: python demo_workflow.py")
            print("2. Or try: python demo_workflow.py interactive")
            print("3. Check the README.md for more examples")
        else:
            print("\n⚠️  Setup completed but some services failed tests")
            print("Check your API keys and network connection")
    else:
        print("\n⚠️  Setup completed but API keys need configuration")
        print("Please edit the .env file with your actual API keys:")
        print("- Weaviate: https://console.weaviate.cloud")
        print("- Cohere: https://cohere.ai")
        print("- Anthropic: https://console.anthropic.com")
        print("\nThen run: python setup.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        sys.exit(1)
