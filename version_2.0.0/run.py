#!/usr/bin/env python
import os
import subprocess
import threading
import time
import webbrowser
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Determine project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SERVER_DIR = PROJECT_ROOT / "server"

def run_backend():
    """Run the FastAPI backend server."""
    try:
        os.chdir(SERVER_DIR)
        logger.info("Starting backend server...")
        
        # Check if this is the first run
        if not (SERVER_DIR / "uploads").exists():
            logger.info("Creating required directories...")
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)
        
        # Run the server
        subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except Exception as e:
        logger.error(f"Backend server error: {str(e)}", exc_info=True)

def run_frontend():
    """Run the frontend development server."""
    try:
        os.chdir(FRONTEND_DIR)
        logger.info("Starting frontend development server...")
        
        # Check if node_modules directory exists
        if not (FRONTEND_DIR / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            if os.name == 'nt':  # Windows
                subprocess.run(["npm", "install"], shell=True)
            else:
                subprocess.run(["npm", "install"])
        
        # Run the frontend
        if os.name == 'nt':  # Windows
            subprocess.run(["npm", "run", "dev"], shell=True)
        else:
            subprocess.run(["npm", "run", "dev"])
    except Exception as e:
        logger.error(f"Frontend server error: {str(e)}", exc_info=True)

def build_frontend():
    """Build the frontend for production."""
    try:
        os.chdir(FRONTEND_DIR)
        logger.info("Building frontend for production...")
        
        # Check if node_modules directory exists
        if not (FRONTEND_DIR / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            if os.name == 'nt':  # Windows
                subprocess.run(["npm", "install"], shell=True)
            else:
                subprocess.run(["npm", "install"])
        
        # Build the frontend
        if os.name == 'nt':  # Windows
            subprocess.run(["npm", "run", "build"], shell=True)
        else:
            subprocess.run(["npm", "run", "build"])
            
        logger.info("Frontend build complete.")
    except Exception as e:
        logger.error(f"Frontend build error: {str(e)}", exc_info=True)
        
def open_browser():
    """Open the browser to the application URL."""
    time.sleep(5)  # Give servers time to start
    webbrowser.open("http://localhost:5173")  # Vite default port

def run_dev():
    """Run both backend and frontend in development mode."""
    # Create threads for backend and frontend
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    browser_thread = threading.Thread(target=open_browser)
    
    # Start the threads
    backend_thread.start()
    frontend_thread.start()
    browser_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")

def run_prod():
    """Build frontend and run backend for production."""
    # Build the frontend first
    build_frontend()
    
    # Run the backend server
    run_backend()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DocuMorph AI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "prod"],
        default="dev",
        help="Run in development or production mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "dev":
        run_dev()
    else:
        run_prod() 