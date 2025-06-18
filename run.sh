#!/bin/bash
set -e

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values if not set in .env
HOST=${HOST:-localhost}
PORT=${PORT:-8007}
MAIN_MODEL_BASE_URL=${MAIN_MODEL_BASE_URL:-${MODEL_BASE_URL:-http://localhost:11434/v1/chat/completions}}
SUB_AGENT_MODEL_BASE_URL=${SUB_AGENT_MODEL_BASE_URL:-${MAIN_MODEL_BASE_URL}}

echo "🤖 Starting AI Agent System with Sub-Agent Spawning..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Display current configuration
echo ""
echo "📊 Current Configuration:"
echo "   Main Agent Model: ${MAIN_MODEL_NAME:-${MODEL_NAME:-llama3:8b}}"
echo "   Main Agent URL: ${MAIN_MODEL_BASE_URL}"
echo "   Sub-Agent Model: ${SUB_AGENT_MODEL_NAME:-${MAIN_MODEL_NAME:-${MODEL_NAME:-llama3:8b}}}"
echo "   Sub-Agent URL: ${SUB_AGENT_MODEL_BASE_URL}"
echo "   Max Loops: ${MAX_LOOP_COUNT:-20}"
echo ""

# Check if LLM servers are running
echo "🔍 Checking LLM server connectivity..."

# Check main model server
if curl -s --connect-timeout 5 "$MAIN_MODEL_BASE_URL" > /dev/null 2>&1; then
    echo "✅ Main agent server accessible at $MAIN_MODEL_BASE_URL"
else
    echo "⚠️ Main agent server not accessible at $MAIN_MODEL_BASE_URL"
fi

# Check sub-agent model server (if different)
if [ "$SUB_AGENT_MODEL_BASE_URL" != "$MAIN_MODEL_BASE_URL" ]; then
    if curl -s --connect-timeout 5 "$SUB_AGENT_MODEL_BASE_URL" > /dev/null 2>&1; then
        echo "✅ Sub-agent server accessible at $SUB_AGENT_MODEL_BASE_URL"
    else
        echo "⚠️ Sub-agent server not accessible at $SUB_AGENT_MODEL_BASE_URL"
    fi
fi

echo ""
echo "💡 Pro tip: Using different models optimizes cost and performance:"
echo "   - Main agent: Handles complex reasoning and coordination"
echo "   - Sub-agents: Execute focused, simple tasks efficiently"
echo ""

# Check if required files exist
required_files=("webapp.py" "functions/core_callable_fns.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        echo "   Please run ./setup.sh first"
        exit 1
    fi
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Display configuration
echo ""
echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Model URL: $MODEL_BASE_URL"
echo "   Logs: ./logs/"
echo ""

# Check for any background processes that might conflict
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️ Port $PORT is already in use. Attempting to stop existing process..."
    pkill -f "uvicorn.*webapp:app" || true
    sleep 2
fi

echo "🚀 Starting web server..."
echo "   Web interface: http://$HOST:$PORT"
echo "   WebSocket endpoint: ws://$HOST:$PORT/ws"
echo ""
echo "💡 Usage tips:"
echo "   • Try asking: 'Find all Python files in the functions directory'"
echo "   • Or: 'Search for FastAPI configuration in the codebase'"
echo "   • Use sub-agents for focused tasks under 200 characters"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down server..."
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the application
exec python webapp.py

# Alternative using uvicorn directly:
# exec uvicorn webapp:app --host "$HOST" --port "$PORT" --reload
