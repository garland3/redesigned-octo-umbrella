#!/usr/bin/env python3
"""
Configuration validation script for AI Agent with Sub-Agent Spawning.
Tests model connectivity and validates configuration.
"""

import os

# Load configuration
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "llama-3.3-70b-versatile")
MAIN_MODEL_BASE_URL = os.getenv("MAIN_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
SUB_AGENT_MODEL_NAME = os.getenv("SUB_AGENT_MODEL_NAME", "llama-3.1-8b-instant")
SUB_AGENT_MODEL_BASE_URL = os.getenv("SUB_AGENT_MODEL_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# API Key Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAIN_MODEL_API_KEY = os.getenv("MAIN_MODEL_API_KEY", GROQ_API_KEY)
SUB_AGENT_MODEL_API_KEY = os.getenv("SUB_AGENT_MODEL_API_KEY", GROQ_API_KEY)

# For backward compatibility
if not MAIN_MODEL_API_KEY and not SUB_AGENT_MODEL_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAIN_MODEL_API_KEY = OPENAI_API_KEY
    SUB_AGENT_MODEL_API_KEY = OPENAI_API_KEY

import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", os.getenv("MODEL_NAME", "llama3:8b"))
MAIN_MODEL_BASE_URL = os.getenv("MAIN_MODEL_BASE_URL", os.getenv("MODEL_BASE_URL", "http://localhost:11434/v1/chat/completions"))
MAIN_MODEL_API_KEY = os.getenv("MAIN_MODEL_API_KEY", os.getenv("OPENAI_API_KEY", ""))

SUB_AGENT_MODEL_NAME = os.getenv("SUB_AGENT_MODEL_NAME", MAIN_MODEL_NAME)
SUB_AGENT_MODEL_BASE_URL = os.getenv("SUB_AGENT_MODEL_BASE_URL", MAIN_MODEL_BASE_URL)
SUB_AGENT_MODEL_API_KEY = os.getenv("SUB_AGENT_MODEL_API_KEY", MAIN_MODEL_API_KEY)

async def test_model_connection(model_name: str, base_url: str, api_key: str, agent_type: str):
    """Test connection to a model API."""
    print(f"\n🧪 Testing {agent_type} connection...")
    print(f"   Model: {model_name}")
    print(f"   URL: {base_url}")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Not set'}")
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Simple test payload
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello! This is a connection test. Please respond with 'OK'."}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Try to extract response content
                    content = "Unknown"
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        content = message.get("content", "No content")
                    
                    print(f"   ✅ Connection successful!")
                    print(f"   Response: {content[:50]}...")
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ HTTP {response.status}: {error_text[:100]}...")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"   ❌ Timeout: Server took too long to respond")
        return False
    except aiohttp.ClientConnectorError:
        print(f"   ❌ Connection failed: Cannot reach server")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def check_environment():
    """Check environment configuration."""
    print("🔧 Environment Configuration Check")
    print("=" * 50)
    
    config_items = [
        ("Main Model Name", MAIN_MODEL_NAME),
        ("Main Model URL", MAIN_MODEL_BASE_URL),
        ("Main Model API Key", "Set" if MAIN_MODEL_API_KEY else "Not set"),
        ("Sub-Agent Model Name", SUB_AGENT_MODEL_NAME),
        ("Sub-Agent Model URL", SUB_AGENT_MODEL_BASE_URL),
        ("Sub-Agent Model API Key", "Set" if SUB_AGENT_MODEL_API_KEY else "Not set"),
    ]
    
    for item, value in config_items:
        status = "✅" if value and value != "Not set" else "⚠️"
        print(f"{status} {item}: {value}")
    
    # Check for common issues
    issues = []
    
    if MAIN_MODEL_BASE_URL == SUB_AGENT_MODEL_BASE_URL and MAIN_MODEL_NAME == SUB_AGENT_MODEL_NAME:
        issues.append("Main and sub-agent using same model (this is fine, but you might want different models)")
    
    if "openai.com" in MAIN_MODEL_BASE_URL and not MAIN_MODEL_API_KEY:
        issues.append("OpenAI URL detected but no API key set for main model")
    
    if "openai.com" in SUB_AGENT_MODEL_BASE_URL and not SUB_AGENT_MODEL_API_KEY:
        issues.append("OpenAI URL detected but no API key set for sub-agent model")
    
    if issues:
        print("\n⚠️ Potential Issues:")
        for issue in issues:
            print(f"   - {issue}")

def suggest_configurations():
    """Suggest optimal configurations."""
    print("\n💡 Recommended Configurations")
    print("=" * 50)
    
    configs = [
        {
            "name": "🏆 Hybrid (Best Performance/Cost)",
            "main": "gpt-4o-mini (OpenAI)",
            "sub": "llama3:8b (Local Ollama)",
            "pros": "Smart coordination + fast execution + cost efficient"
        },
        {
            "name": "💰 Budget Cloud",
            "main": "gpt-3.5-turbo (OpenAI)",
            "sub": "gpt-3.5-turbo (OpenAI)",
            "pros": "Predictable costs + reliable performance"
        },
        {
            "name": "🔒 Fully Local",
            "main": "llama3:70b (Local)",
            "sub": "llama3:8b (Local)",
            "pros": "No API costs + complete privacy + offline"
        },
        {
            "name": "🌐 Cross-Platform",
            "main": "claude-3-haiku (Anthropic)",
            "sub": "gpt-3.5-turbo (OpenAI)",
            "pros": "Provider diversity + specialized strengths"
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"   Main Agent: {config['main']}")
        print(f"   Sub-Agents: {config['sub']}")
        print(f"   Benefits: {config['pros']}")

async def main():
    """Main validation function."""
    print("🚀 AI Agent Configuration Validator")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Test connections
    print(f"\n🌐 Connection Tests")
    print("=" * 50)
    
    main_success = await test_model_connection(
        MAIN_MODEL_NAME, MAIN_MODEL_BASE_URL, MAIN_MODEL_API_KEY, "Main Agent"
    )
    
    # Only test sub-agent if different from main
    if (SUB_AGENT_MODEL_BASE_URL != MAIN_MODEL_BASE_URL or 
        SUB_AGENT_MODEL_NAME != MAIN_MODEL_NAME):
        sub_success = await test_model_connection(
            SUB_AGENT_MODEL_NAME, SUB_AGENT_MODEL_BASE_URL, SUB_AGENT_MODEL_API_KEY, "Sub-Agent"
        )
    else:
        print(f"\n🔄 Sub-agent using same configuration as main agent")
        sub_success = main_success
    
    # Summary
    print(f"\n📊 Summary")
    print("=" * 50)
    
    if main_success and sub_success:
        print("✅ All systems ready! You can start the agent with ./run.sh")
    else:
        print("❌ Some connections failed. Please check your configuration.")
        print("   - Verify API keys are correct")
        print("   - Ensure servers are running (e.g., ollama serve)")
        print("   - Check firewall settings")
    
    # Show configuration suggestions
    suggest_configurations()
    
    print(f"\n🔧 To fix issues:")
    print("   1. Edit .env file with correct values")
    print("   2. Start required servers (e.g., ollama serve)")
    print("   3. Run this script again to verify")

if __name__ == "__main__":
    asyncio.run(main())
