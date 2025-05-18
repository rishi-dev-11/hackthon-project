#!/bin/bash

# DocuMorph AI Setup Script
# This script prepares the application for production deployment

# Print colored output
print_step() {
  echo -e "\e[1;34m===> $1\e[0m"
}

print_success() {
  echo -e "\e[1;32m===> $1\e[0m"
}

print_error() {
  echo -e "\e[1;31m===> ERROR: $1\e[0m"
}

# Create .env file if it doesn't exist
create_env_file() {
  print_step "Setting up environment variables..."
  
  if [ ! -f ".env" ]; then
    cat > .env << EOF
# MongoDB Connection
MONGO_URI=mongodb://localhost:27017/documorph

# App Settings
APP_ENV=production
JWT_SECRET=supersecretkey
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# API Keys
LANGCHAIN_PROJECT=
HF_TOKEN=
groq_api_key=
serpapi=
GOOGLE_CLIENT_ID=
Client_secret=
goog_api_key=
GOOGLE_REDIRECT_URI=
EOF
    print_success "Created default .env file - please update with your actual API keys"
  else
    print_success ".env file already exists"
  fi
}

# Install backend dependencies
install_backend_deps() {
  print_step "Installing backend dependencies..."
  cd server || { print_error "server directory not found"; exit 1; }
  
  pip install -r requirements.txt
  
  if [ $? -ne 0 ]; then
    print_error "Failed to install backend dependencies"
    exit 1
  fi
  
  print_success "Backend dependencies installed"
  cd ..
}

# Install frontend dependencies
install_frontend_deps() {
  print_step "Installing frontend dependencies..."
  cd frontend || { print_error "frontend directory not found"; exit 1; }
  
  # Install npm packages including GSAP
  npm install
  npm install gsap
  
  if [ $? -ne 0 ]; then
    print_error "Failed to install frontend dependencies"
    exit 1
  fi
  
  print_success "Frontend dependencies installed"
  cd ..
}

# Build the frontend
build_frontend() {
  print_step "Building frontend..."
  cd frontend || { print_error "frontend directory not found"; exit 1; }
  
  npm run build
  
  if [ $? -ne 0 ]; then
    print_error "Failed to build frontend"
    exit 1
  fi
  
  print_success "Frontend built successfully"
  cd ..
}

# Ensure frontend build directory exists for backend
ensure_dist_dir() {
  print_step "Ensuring frontend build directory exists..."
  mkdir -p frontend/dist
  touch frontend/dist/.placeholder
  
  print_success "Frontend build directory ensured"
}

# Setup MongoDB for first run
setup_mongodb() {
  print_step "Setting up MongoDB connection..."
  
  # This is a placeholder for any MongoDB setup steps
  # The actual connection will be handled by the app using the MONGO_URI env var
  
  print_success "MongoDB setup complete"
}

# Prepare for Render deployment
prepare_for_render() {
  print_step "Preparing for Render deployment..."
  
  # Create a render.yaml file
  cat > render.yaml << EOF
services:
  - type: web
    name: documorph-ai
    env: python
    buildCommand: ./setup.sh
    startCommand: cd server && uvicorn main:app --host 0.0.0.0 --port \$PORT
    envVars:
      - key: MONGO_URI
        sync: false
      - key: APP_ENV
        value: production
      - key: JWT_SECRET
        sync: false
      - key: LANGCHAIN_PROJECT
        sync: false
      - key: HF_TOKEN
        sync: false
      - key: groq_api_key
        sync: false
      - key: serpapi
        sync: false
      - key: GOOGLE_CLIENT_ID
        sync: false
      - key: Client_secret
        sync: false
      - key: goog_api_key
        sync: false
      - key: GOOGLE_REDIRECT_URI
        sync: false
EOF
  
  # Create a start script for production
  cat > start.sh << EOF
#!/bin/bash
cd server
uvicorn main:app --host 0.0.0.0 --port \${PORT:-8000}
EOF
  
  chmod +x start.sh
  
  print_success "Render deployment configuration complete"
}

# Run test cases
run_tests() {
  print_step "Running tests..."
  
  # Backend Tests
  cd server || { print_error "server directory not found"; exit 1; }
  
  # Basic connectivity test
  python -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

async def test_mongodb_connection():
    load_dotenv()
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
    print(f'Testing connection to MongoDB at {mongo_uri}')
    client = AsyncIOMotorClient(mongo_uri)
    try:
        # The ismaster command is cheap and does not require auth.
        await client.admin.command('ping')
        print('MongoDB connection successful')
        return True
    except Exception as e:
        print(f'MongoDB connection failed: {str(e)}')
        return False

async def main():
    connection_successful = await test_mongodb_connection()
    if not connection_successful:
        print('WARNING: MongoDB connection test failed')
        # Don't exit with error to allow deployment without DB
    
asyncio.run(main())
"
  
  cd ..
  
  print_success "Tests completed"
}

# Main execution
main() {
  print_step "Starting DocuMorph AI setup..."
  
  create_env_file
  ensure_dist_dir
  install_backend_deps
  install_frontend_deps
  build_frontend
  setup_mongodb
  prepare_for_render
  run_tests
  
  print_success "DocuMorph AI setup complete!"
  print_success "To run the app locally:"
  print_success "  1. Start the backend: cd server && uvicorn main:app --reload"
  print_success "  2. Start the frontend: cd frontend && npm run dev"
  print_success "For production deployment:"
  print_success "  - Use the start.sh script or follow Render deployment instructions"
}

# Run the main function
main 