#!/bin/bash
# CLI Agent Runner Script
# Usage: ./run_cli.sh [prompt]
# Default prompt: "tell me about this dir"

cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    echo "Running CLI agent with default prompt: 'tell me about this dir'"
    python3 cli_agent.py
else
    echo "Running CLI agent with prompt: '$*'"
    python3 cli_agent.py "$@"
fi
