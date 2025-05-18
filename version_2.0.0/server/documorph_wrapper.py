"""
DocuMorph Wrapper Module - Imports key functions from backend modules
"""
import os
import sys
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add all potential directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(parent_dir, "backend")
project_root = os.path.dirname(parent_dir)

# Add all paths to sys.path to ensure imports work
for path in [current_dir, parent_dir, backend_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

logger.info(f"Import paths: {sys.path}")

# Define required functions
REQUIRED_FUNCTIONS = [
    "process_document",
    "apply_template_to_document",
    "initialize_db_templates",
    "get_templates_for_user",
    "extract_document_structure", 
    "detect_tables_from_pdf",
    "detect_figures_from_pdf"
]

# First try to import from server modules directory
try:
    # Try importing from local modules directory first
    from modules.documorph_ai import UserTier
    module_path = "modules.documorph_ai"
    logger.info(f"Imported UserTier from {module_path}")
    
    # Import all required functions
    for func_name in REQUIRED_FUNCTIONS:
        try:
            exec(f"from modules.documorph_ai import {func_name}")
            logger.info(f"Imported {func_name} from {module_path}")
        except ImportError as e:
            logger.warning(f"Could not import {func_name} from {module_path}: {e}")
            
except ImportError:
    logger.warning("Could not import from modules.documorph_ai, trying backend.documorph_ai")
    
    # Try importing from backend directory
    try:
        from backend.documorph_ai import UserTier
        module_path = "backend.documorph_ai"
        logger.info(f"Imported UserTier from {module_path}")
        
        # Import all required functions
        for func_name in REQUIRED_FUNCTIONS:
            try:
                exec(f"from backend.documorph_ai import {func_name}")
                logger.info(f"Imported {func_name} from {module_path}")
            except ImportError as e:
                logger.warning(f"Could not import {func_name} from {module_path}: {e}")
                
    except ImportError:
        logger.error("Could not import from backend.documorph_ai")
        
        # Define fallback UserTier class
        class UserTier:
            FREE = "free"
            PREMIUM = "premium"
            
            @staticmethod
            def get_tier_features():
                return {
                    UserTier.FREE: ["basic_processing", "tables"],
                    UserTier.PREMIUM: ["basic_processing", "tables", "advanced_nlp", "translation"]
                }

# Import auxiliary modules for table and chart extraction
try:
    from modules.table_extraction import extract_tables
    from modules.figure_extraction import extract_images_from_pdf
    logger.info("Successfully imported table and figure extraction modules")
except ImportError:
    logger.warning("Could not import table and figure extraction modules")
    
    # Define fallback functions
    def extract_tables(*args, **kwargs):
        logger.warning("Using fallback extract_tables function")
        return []
        
    def extract_images_from_pdf(*args, **kwargs):
        logger.warning("Using fallback extract_images_from_pdf function")
        return []

# Define fallback functions for any missing required functions
for func_name in REQUIRED_FUNCTIONS:
    if func_name not in globals():
        logger.warning(f"Creating fallback for missing function: {func_name}")
        exec(f"""
def {func_name}(*args, **kwargs):
    logger.error("Using fallback {func_name} function - real implementation not available")
    return {{}} if '{func_name}' == 'process_document' else None
""")

# Export all imported functions and classes
__all__ = [
    "UserTier",
    "process_document",
    "apply_template_to_document", 
    "initialize_db_templates",
    "get_templates_for_user",
    "extract_document_structure",
    "detect_tables_from_pdf",
    "detect_figures_from_pdf",
    "extract_tables",
    "extract_images_from_pdf"
]

logger.info("DocuMorph wrapper module initialized")
