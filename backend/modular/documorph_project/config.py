import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Typically for non-OAuth Google services
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE") # Ensure this is set

# Persona Categories (for LLM interaction style)
PERSONA_CATEGORIES = [
    "Student", "Content Creator", "Researcher", "Business Professional",
    "Multilingual User", "Author", "Collaborator", "Project Manager", "General/Others"
]

# User tier system
class UserTier:
    FREE = "free"
    PREMIUM = "premium"

    @staticmethod
    def get_tier_features():
        return {
            UserTier.FREE: {
                "max_documents_processed": 5, # Max docs user can process and store metadata for
                "max_custom_templates": 3,    # Max formatting templates user can save
                "llm_persona_guidance": False,# LLM uses generic prompts
                "llm_advanced_features": False, # e.g., complex style enhancement, deep structure analysis
                "multi_language_ocr": ["eng"],# Limited OCR languages
                "multi_language_translation": False, # For UI elements or content hints
                "style_guide_compliance_check": False,
                "async_processing": False,
                "team_collaboration_features": False,
                "advanced_table_formatting": False,
                "advanced_caption_editor": False,
                "google_drive_export": False,
                "ms_word_integration_features": False, # e.g. add-in
                "plagiarism_check": False,
                "allowed_personas_for_llm": ["Student", "Content Creator", "General/Others"], # Limited persona choice
                "pdf_export_quality": "basic", # Basic PDF export
                "image_export_feature": True, # Exporting all images from PDF
                "custom_formatting_templates": True # Can create own formatting templates
            },
            UserTier.PREMIUM: {
                "max_documents_processed": 100,
                "max_custom_templates": 50,
                "llm_persona_guidance": True, # Can select from all personas for LLM
                "llm_advanced_features": True,
                "multi_language_ocr": ["eng", "fra", "deu", "spa", "rus", "ara", "chi_sim", "jpn", "kor", "hi"],
                "multi_language_translation": True,
                "style_guide_compliance_check": True,
                "async_processing": True,
                "team_collaboration_features": True,
                "advanced_table_formatting": True,
                "advanced_caption_editor": True,
                "google_drive_export": True,
                "ms_word_integration_features": True,
                "plagiarism_check": True,
                "allowed_personas_for_llm": PERSONA_CATEGORIES, # All personas
                "pdf_export_quality": "enhanced", # Better PDF export
                "image_export_feature": True,
                "custom_formatting_templates": True
            }
        }

# Logging configuration
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add a handler to ensure logs are displayed in Streamlit's console during development
    # For production, you'd use more robust logging (e.g., to a file or logging service)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # logging.getLogger().addHandler(stream_handler) # Add to root logger
    return logging.getLogger("DocuMorphAI") # Specific logger name

logger = setup_logging()

# Constants for directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is in the project root
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "storage", "vector_stores")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "storage", "user_documents") # For persisted user uploads
OUTPUT_DIR = os.path.join(BASE_DIR, "storage", "formatted_outputs")
TEMP_DIR = os.path.join(BASE_DIR, "storage", "temp") # For truly temporary files

# Ensure these directories exist
for dir_path in [VECTOR_STORE_DIR, DOCUMENTS_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Availability flags (checked where used)
try:
    import google.oauth2.credentials
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

try:
    import msoffcrypto
    MSOFFCRYPTO_AVAILABLE = True
except ImportError:
    MSOFFCRYPTO_AVAILABLE = False

try:
    import unstructured
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# Flag for your conceptual 'documorph_fixes' module
# If this module is not strictly necessary after refactoring, this can be removed.
try:
    import documorph_fixes # Assuming this would be a .py file you create
    FIXES_MODULE_AVAILABLE = True
except ImportError:
    FIXES_MODULE_AVAILABLE = False
    if os.path.exists(os.path.join(BASE_DIR, "documorph_fixes.py")): # Check if file exists but import failed
        logger.error("documorph_fixes.py exists but could not be imported. Check for syntax errors or missing dependencies within it.")


# ReportLab availability check (for PDF generation)
try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation capabilities will be limited. `pip install reportlab`")

# WeasyPrint availability check (for potentially higher fidelity HTML->PDF)
try:
    # WeasyPrint depends on external GTK+ libraries which might not be installed
    # Specifically, it's looking for libgobject-2.0-0.dll in Tesseract-OCR
    try:
        import weasyprint
        WEASYPRINT_AVAILABLE = True
    except OSError as e:
        # Catch the specific OSError related to missing DLLs
        WEASYPRINT_AVAILABLE = False
        logger.warning(f"WeasyPrint found but cannot load required library: {e}. Enhanced PDF generation will be unavailable.")
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint not installed. Enhanced PDF generation from HTML will be unavailable. `pip install WeasyPrint`")
except Exception as e:
    WEASYPRINT_AVAILABLE = False
    logger.warning(f"Unexpected error loading WeasyPrint: {e}. Enhanced PDF generation will be unavailable.")