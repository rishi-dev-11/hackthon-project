#!/bin/bash

# DocuMorph AI Installation Script
echo "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛"
echo "⬛                                                               ⬛"
echo "⬛           DocuMorph AI Installation Script                   ⬛"
echo "⬛                                                               ⬛"
echo "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛"
echo ""

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 8 ]]; then
    echo "❌ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Found Python $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment. Please install venv package."
    exit 1
fi
echo "✅ Created virtual environment"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment."
    exit 1
fi
echo "✅ Activated virtual environment"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "❌ Failed to upgrade pip."
    exit 1
fi
echo "✅ Upgraded pip"

# Install requirements
echo "Installing requirements (this may take a while)..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements."
    exit 1
fi
echo "✅ Installed requirements"

# Install spaCy model
echo "Installing spaCy model..."
python -m spacy download en_core_web_sm
if [ $? -ne 0 ]; then
    echo "❌ Failed to download spaCy model."
    exit 1
fi
echo "✅ Installed spaCy model"

# Check MongoDB
echo "Checking for MongoDB..."
which mongod > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ MongoDB not found. Some features may not work without MongoDB."
    echo "   Please install MongoDB if you want to use all features."
else
    echo "✅ MongoDB found"
fi

echo ""
echo "✨ Installation complete! ✨"
echo ""
echo "To start DocuMorph AI, run:"
echo "source venv/bin/activate"
echo "cd backend"
echo "streamlit run documorph_ai.py"
echo ""
echo "⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛" 