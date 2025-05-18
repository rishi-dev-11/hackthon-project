# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("serpapi")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GOOGLE_API_KEY = os.getenv("GOOG_API_KEY")

# User tier system
class UserTier:
    FREE = "free"
    PREMIUM = "premium"
    
    @staticmethod
    def get_tier_features():
        return {
            UserTier.FREE: {
                "max_documents": 5,
                "max_templates": 3,
                "llm_enabled": False,
                "multi_language": False,
                "ocr_languages": ["eng"],
                "style_guide_compliance": False,
                "async_processing": False,
                "team_collaboration": False,
                "advanced_tables": False,
                "caption_editor": False,
                "google_docs": False,
                "ms_word": False,
                "plagiarism_check": False,
                "template_categories": ["Student", "Content Creator","Others"],
                "google_drive_export": False,
                "custom_templates": True
            },
            UserTier.PREMIUM: {
                "max_documents": 100,
                "max_templates": 50,
                "llm_enabled": True,
                "multi_language": True,
                "ocr_languages": ["eng", "fra", "deu", "spa", "rus", "ara", "chi_sim", "jpn", "kor","hi"],
                "style_guide_compliance": True,
                "async_processing": True,
                "team_collaboration": True,
                "advanced_tables": True,
                "caption_editor": True,
                "google_docs": True,
                "ms_word": True,
                "plagiarism_check": True,
                "template_categories": ["Student", "Content Creator", "Researcher", "Business Professional", "Multilingual User", "Author", "Collaborator", "Project Manager"],
                "google_drive_export": True,
                "custom_templates": True
            }
        }

# Availability flags (can be checked at runtime in main app or modules)
GOOGLE_API_AVAILABLE = True # Assume true, main app will warn if import fails
MSOFFCRYPTO_AVAILABLE = True # Assume true, main app will warn if import fails
UNSTRUCTURED_AVAILABLE = True # Assume true, main app will warn if import fails
FIXES_AVAILABLE = True # Assume true, main app will warn if import fails
