#!/bin/bash
# Run Market Dashboard with virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the dashboard
streamlit run dashboard/app.py "$@"
