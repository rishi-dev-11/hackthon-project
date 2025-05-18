#!/usr/bin/env bash
# build.sh for render.com deployment

# Exit on error
set -o errexit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Create any necessary directories
mkdir -p documents
mkdir -p output
mkdir -p vector_stores 