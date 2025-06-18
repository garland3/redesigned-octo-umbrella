#!/bin/bash
set -e

echo "🚀 Setting up AI Agent System with Sub-Agent Spawning..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Create virtual environment with uv
echo "📦 Creating virtual environment..."
uv venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "📝 Creating requirements.txt..."
    cat > requirements.txt << EOF
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
fastmcp>=0.2.0
pathlib2>=2.3.0
EOF
fi

# Install dependencies with uv
echo "📥 Installing dependencies with uv..."
uv pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating default .env configuration..."
    cat > .env << EOF
# Main Agent Configuration (Coordinator)
# Use a more powerful model for complex reasoning and coordination
MAIN_MODEL_NAME=gpt-4o-mini
MAIN_MODEL_BASE_URL=https://api.openai.com/v1/chat/completions
MAIN_MODEL_API_KEY=your_openai_api_key_here

# Sub-Agent Configuration (Focused Tasks)
# Use a less powerful/cheaper model for simple, focused tasks
SUB_AGENT_MODEL_NAME=llama3:8b
SUB_AGENT_MODEL_BASE_URL=http://localhost:11434/v1/chat/completions
SUB_AGENT_MODEL_API_KEY=

# General Configuration
MAX_LOOP_COUNT=20
MAX_CONTEXT_CHARACTERS=15000

# Legacy environment variables (for backwards compatibility)
MODEL_NAME=llama3:8b
MODEL_BASE_URL=http://localhost:11434/v1/chat/completions
MODEL_API_KEY=
EOF
    echo "📝 Created .env file with dual-model configuration"
    echo "   - Edit MAIN_MODEL_* for your coordination agent"
    echo "   - Edit SUB_AGENT_MODEL_* for your focused sub-agents"

# Agent Configuration
MAX_LOOP_COUNT=20
MAX_CONTEXT_CHARACTERS=15000

# Server Configuration
HOST=localhost
PORT=8007
EOF
    echo "📄 Created .env file with default settings"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p static
mkdir -p functions

# Check if all required files exist
echo "🔍 Checking required files..."
required_files=(
    "webapp.py"
    "functions/core_callable_fns.py"
    "functions/intellegent_file_search.py"
    "functions/file_system_fns.py"
    "functions/coerce_to_json.py"
    "functions/helper.py"
    "static/index.html"
    "static/overview.rst"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ All required files are present"
else
    echo "⚠️ Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
fi

# Test the installation
echo "🧪 Testing installation..."
python -c "
import fastapi
import uvicorn
import websockets
import aiohttp
import dotenv
print('✅ All dependencies imported successfully')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Model Configuration Options:"
echo "   1. 🏆 Hybrid Setup (Recommended):"
echo "      Main: GPT-4/Claude (smart coordination)"
echo "      Sub: Local Ollama (fast, cheap tasks)"
echo ""
echo "   2. 💰 Cost-Effective:"
echo "      Main: GPT-4o-mini (good reasoning)"
echo "      Sub: GPT-3.5-turbo (reliable tasks)"
echo ""
echo "   3. 🔒 Fully Local:"
echo "      Main: llama3:70b (best local reasoning)"
echo "      Sub: llama3:8b (fast local tasks)"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file to configure your API keys and models"
echo "   2. Run: ./run.sh"
echo "   3. Open browser: http://localhost:8007"
echo ""
echo "💡 The system automatically uses different models for different purposes:"
echo "   - Main agent handles complex coordination and planning"
echo "   - Sub-agents handle focused, simple tasks efficiently"
echo ""
