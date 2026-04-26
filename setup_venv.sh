#!/usr/bin/env bash
# Setup virtual environment for Early_exit_architecture_research
# Run this script from the project root.

# Fail on any error
set -e

# Create virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
source "venv/bin/activate"

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Print success message
echo "✅ Virtual environment is ready. Use 'source venv/bin/activate' to activate it."
