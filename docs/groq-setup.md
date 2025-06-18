# Groq Configuration Guide

## Getting Started with Groq

1. **Get a Groq API Key**:
   - Visit https://console.groq.com/
   - Sign up for a free account
   - Generate an API key from the dashboard

2. **Configure Your .env File**:
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

## Recommended Model Combinations

### Best Performance (Groq + Groq)
- **Main Agent**: `llama-3.1-70b-versatile` - Smart coordination and complex reasoning
- **Sub-Agents**: `llama-3.1-8b-instant` - Fast execution of simple tasks

### Hybrid Setup (Groq + Local)
- **Main Agent**: `llama-3.1-70b-versatile` - Smart coordination via Groq
- **Sub-Agents**: `llama3:8b` - Local Ollama for privacy/cost savings

### Cost-Optimized (Local + Local)
- **Main Agent**: `llama3:70b` - Local Ollama 70B model
- **Sub-Agents**: `llama3:8b` - Local Ollama 8B model

## Available Groq Models

### For Main Agent (Complex Tasks)
- `llama-3.1-70b-versatile` - Best reasoning, slower
- `llama-3.1-8b-instant` - Balanced speed/performance
- `mixtral-8x7b-32768` - Good for large context
- `gemma2-9b-it` - Alternative option

### For Sub-Agents (Simple Tasks)
- `llama-3.1-8b-instant` - Recommended for sub-agents
- `gemma-7b-it` - Lightweight alternative
- `llama3-8b-8192` - Standard option

## Configuration Examples

### Example 1: Pure Groq Setup
```bash
MAIN_MODEL_NAME=llama-3.1-70b-versatile
MAIN_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions

SUB_AGENT_MODEL_NAME=llama-3.1-8b-instant
SUB_AGENT_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions

GROQ_API_KEY=gsk_your_api_key_here
```

### Example 2: Groq Main + Local Sub-Agents
```bash
MAIN_MODEL_NAME=llama-3.1-70b-versatile
MAIN_MODEL_BASE_URL=https://api.groq.com/openai/v1/chat/completions
GROQ_API_KEY=gsk_your_api_key_here

SUB_AGENT_MODEL_NAME=llama3:8b
SUB_AGENT_MODEL_BASE_URL=http://localhost:11434/v1/chat/completions
```
```

## Benefits of This Configuration

### Why 70B for Main Agent?
- Better at understanding complex requests
- Superior task decomposition and coordination
- More reliable at following system prompts
- Better context understanding

### Why 8B for Sub-Agents?
- Faster execution for simple, focused tasks
- Lower cost per request
- Sufficient capability for well-defined tasks
- Reduced latency for quick operations

## Cost Considerations

### Groq Pricing (as of 2024)
- **Llama 3.1 70B**: ~$0.59/1M input tokens, ~$0.79/1M output tokens
- **Llama 3.1 8B**: ~$0.05/1M input tokens, ~$0.08/1M output tokens

### Cost Optimization Tips
1. Use 8B model for sub-agents (10x cheaper than 70B)
2. Set appropriate context limits to control token usage
3. Use local models for development/testing
4. Monitor usage through Groq console

## Troubleshooting

### Common Issues
1. **Rate Limits**: Groq has generous limits but they exist
2. **API Key Format**: Should start with `gsk_`
3. **Model Names**: Use exact names from Groq documentation
4. **Network**: Ensure internet access for Groq API calls

### Testing Your Setup
Run the validation script:
```bash
python validate_config.py
```

This will test both your main and sub-agent configurations.
