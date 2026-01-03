#!/bin/bash

# Geo-Llama Stable Launch Script
# Use this to show the judges the O(1) Memory capabilities.

# 1. Get the directory of this script to allow running from anywhere
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 2. Print Header
echo "=========================================================="
echo "   Geo-Llama: Hybrid Structural Intelligence Demo"
echo "   (c) 2026 Antigravity Research"
echo "=========================================================="
echo " > Mode: Stable Presentation"
echo " > Device: CPU (Universal Compatibility)"
echo " > Architecture: Cl(4,1) Conformal Geometric Algebra + Llama-3.2"
echo "=========================================================="
echo ""

# 3. Check for Python/Virtualenv
# We assume the user has a valid python environment active or 'python3' is available.
# Ideally, we piggyback off the parent folder's venv if it exists.

PYTHON_EXEC="python3"
if [ -d "../venv" ]; then
    echo "[!] Detected parent virtualenv, using it..."
    PYTHON_EXEC="../venv/bin/python3"
elif [ -d "venv" ]; then
     echo "[!] Detected local virtualenv, using it..."
     PYTHON_EXEC="venv/bin/python3"
fi

# 4. Run the Interface
# We set PYTHONPATH to current dir so 'import geo_llama' works
export PYTHONPATH="$DIR:$PYTHONPATH"

$PYTHON_EXEC chat.py

# 5. Pause on exit so judges can read errors if any
echo ""
echo "[SESSION ENDED] Press Enter to close window..."
read
