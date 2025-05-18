import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory(path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def copy_file(source, destination):
    """Copy a file from source to destination."""
    try:
        shutil.copy2(source, destination)
        logger.info(f"Copied {source} to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error copying {source}: {e}")
        return False

def setup_integrations():
    """Set up integrations by copying necessary files from version1.0.0 to the backend directory."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    version_dir = os.path.join(os.path.dirname(base_dir), "backend", "version1.0.0")
    
    # Check if version1.0.0 directory exists
    if not os.path.exists(version_dir):
        logger.error(f"Version directory not found: {version_dir}")
        return False
    
    # Files to copy
    files_to_copy = [
        "table_extraction.py",
        "chart_extraction.py",
        "documorph_fixes.py"
    ]
    
    # Copy each file
    success = True
    for file in files_to_copy:
        source = os.path.join(version_dir, file)
        destination = os.path.join(base_dir, file)
        
        if not os.path.exists(source):
            logger.warning(f"Source file not found: {source}")
            success = False
            continue
            
        if not copy_file(source, destination):
            success = False
    
    if success:
        logger.info("All integration files copied successfully")
    else:
        logger.warning("Some integration files could not be copied")
    
    return success

def create_init_file():
    """Create __init__.py to make the directory a proper package."""
    init_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
    
    # Only create if it doesn't exist
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# Package initialization file\n")
        logger.info(f"Created __init__.py at {init_path}")
    
    return True

if __name__ == "__main__":
    logger.info("Setting up DocuMorph AI integrations...")
    setup_integrations()
    create_init_file()
    logger.info("Integration setup completed") 