#!/bin/bash

echo "🧪 Testing AI Agent System Setup..."

# Check if setup was run
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate environment
source venv/bin/activate

echo "✅ Virtual environment activated"

# Test imports
echo "🔍 Testing Python imports..."
python -c "
try:
    import fastapi
    import uvicorn
    import websockets
    import aiohttp
    from dotenv import load_dotenv
    print('✅ All core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test custom modules
echo "🔍 Testing custom modules..."
python -c "
import sys
sys.path.append('.')
try:
    from functions.core_callable_fns import spawn_new_agent, all_work_is_finished
    from functions.intellegent_file_search import intelligent_file_search
    from functions.file_system_fns import list_files, read_file
    print('✅ All custom modules imported successfully')
except ImportError as e:
    print(f'❌ Custom module import error: {e}')
    exit(1)
"

# Test basic functionality
echo "🔍 Testing basic functionality..."
python -c "
import sys
sys.path.append('.')
from functions.core_callable_fns import validate_task_for_simple_llm, create_simple_task

# Test task validation
result = validate_task_for_simple_llm('Find authentication in docs')
print(f'Task validation: {result}')

# Test task creation
task = create_simple_task('search_keyword', 'FastAPI', 'in configuration files')
print(f'Task creation: {task}')

print('✅ Basic functionality tests passed')
"

# Check file structure
echo "🔍 Checking file structure..."
required_files=(
    "webapp.py"
    "functions/core_callable_fns.py"
    "functions/intellegent_file_search.py"
    "functions/file_system_fns.py"
    "functions/coerce_to_json.py"
    "functions/helper.py"
    "static/index.html"
    "static/overview.rst"
    "requirements.txt"
    ".env"
    "setup.sh"
    "run.sh"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ All required files present"
else
    echo "❌ Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
fi

# Test configuration
echo "🔍 Testing configuration..."
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    if grep -q "MODEL_NAME" .env && grep -q "PORT" .env; then
        echo "✅ .env file contains required variables"
    else
        echo "⚠️ .env file may be missing some variables"
    fi
else
    echo "❌ .env file missing"
fi

echo ""
echo "🎉 Test complete!"
echo ""
echo "To start the system:"
echo "  ./run.sh"
echo ""
echo "Then open: http://localhost:8007"
