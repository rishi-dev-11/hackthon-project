import streamlit as st
import uuid
import logging
from pymongo import MongoClient, errors as pymongo_errors # Import specific errors
from config import UserTier, PERSONA_CATEGORIES # Import PERSONA_CATEGORIES
import os
logger = logging.getLogger("DocuMorphAI") # Use the app-specific logger

@st.cache_resource # Cache the MongoDB client and collections
def init_mongodb():
    try:
        # Consider adding MONGO_URI to .env for flexibility
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000) # Increased timeout slightly
        client.admin.command('ping') # Verify connection
        db = client["documorph_db"]
        # Consider adding indexes for frequently queried fields like user_id, template_id
        # e.g., db.templates.create_index([("user_id", 1)])
        # e.g., db.templates.create_index([("template_id", 1)], unique=True)
        logger.info(f"MongoDB connection established to {mongo_uri.split('@')[-1] if '@' in mongo_uri else mongo_uri}.")
        return db.documents, db.templates, db.users
    except pymongo_errors.ConnectionFailure as e: # More specific error
        logger.error(f"MongoDB connection failed (ConnectionFailure): {e}", exc_info=True)
        st.error("Critical: Failed to connect to the database. Functionality will be severely limited.")
        return None, None, None
    except Exception as e: # Catch other potential errors during init
        logger.error(f"An unexpected error occurred during MongoDB initialization: {e}", exc_info=True)
        st.error("Critical: An unexpected database error occurred.")
        return None, None, None


def create_default_formatting_templates():
    """Defines a list of default formatting templates with various roles (personas)."""
    templates = [
        {
            "name": "Student Essay (MLA Style)", "is_custom": False, "persona_category": "Student",
            "margin_top": 1.0, "margin_bottom": 1.0, "margin_left": 1.0, "margin_right": 1.0,
            "body_font": "Times New Roman", "body_font_size": 12,
            "heading_font": "Times New Roman", "heading1_font_size": 12, "heading1_bold": True, "heading1_align": "center",
            "heading2_font_size": 12, "heading2_bold": False,
            "heading3_font_size": 12, "heading3_bold": False,
            "line_spacing": 2.0, "paragraph_spacing_after": 0, "first_line_indent": 0.5,
            "header_text": "[Last Name] [PageNumber]", "header_align": "right",
            "footer_text": "",
            "title_page_enabled": False,
            "toc_enabled": False,
            "color_scheme": "Default (Black/White)", "include_tables_figures": True,
            "description": "Standard MLA format for student essays and papers."
        },
        {
            "name": "Business Report - Formal", "is_custom": False, "persona_category": "Business Professional",
            "margin_top": 1.0, "margin_bottom": 1.0, "margin_left": 1.25, "margin_right": 1.0,
            "body_font": "Arial", "body_font_size": 11,
            "heading_font": "Arial", "heading1_font_size": 16, "heading1_bold": True, "heading1_space_after": 12,
            "heading2_font_size": 14, "heading2_bold": True, "heading2_space_after": 6,
            "heading3_font_size": 12, "heading3_bold": True, "heading3_space_after": 6,
            "line_spacing": 1.15, "paragraph_spacing_after": 6, "first_line_indent": 0.0,
            "header_text": "[Company Name] - Confidential", "header_align": "left",
            "footer_text": "Page [PageNumber] of [TotalPages]", "footer_align": "right",
            "title_page_enabled": True, "title_page_title": "[Report Title]", "title_page_subtitle": "[Report Subtitle/Date]",
            "toc_enabled": True, "toc_levels": 3,
            "color_scheme": "Blue Professional", "include_tables_figures": True,
            "description": "A formal template for business reports, proposals, and analyses."
        },
        {
            "name": "Research Paper (APA Style)", "is_custom": False, "persona_category": "Researcher",
            "margin_top": 1.0, "margin_bottom": 1.0, "margin_left": 1.0, "margin_right": 1.0,
            "body_font": "Times New Roman", "body_font_size": 12,
            "heading_font": "Times New Roman",
            "heading1_font_size": 12, "heading1_bold": True, "heading1_align": "center",
            "heading2_font_size": 12, "heading2_bold": True, "heading2_align": "left",
            "heading3_font_size": 12, "heading3_bold": True, "heading3_italic": True, "heading3_align": "left",
            "line_spacing": 2.0, "paragraph_spacing_after": 0, "first_line_indent": 0.5,
            "header_text": "[Running Head - Max 50 Chars]                            [PageNumber]", "header_align": "left", # Manual spacing for right align page
            "footer_text": "",
            "title_page_enabled": True, "title_page_title": "[Paper Title]", "title_page_author": "[Author Name(s)]", "title_page_affiliation": "[Affiliation(s)]",
            "toc_enabled": False, # APA usually doesn't have TOC for articles
            "color_scheme": "Academic", "include_tables_figures": True,
            "description": "APA style template for research articles and scientific papers."
        },
        {
            "name": "Creative Blog Post", "is_custom": False, "persona_category": "Content Creator",
            "margin_top": 1.25, "margin_bottom": 1.25, "margin_left": 1.5, "margin_right": 1.5,
            "body_font": "Georgia", "body_font_size": 12,
            "heading_font": "Helvetica", "heading1_font_size": 24, "heading1_bold": True, "heading1_space_after": 18,
            "heading2_font_size": 18, "heading2_bold": True, "heading2_space_after": 12,
            "heading3_font_size": 14, "heading3_bold": True, "heading3_italic": True, "heading3_space_after": 6,
            "line_spacing": 1.6, "paragraph_spacing_after": 12, "first_line_indent": 0.0,
            "header_text": "",
            "footer_text": "© [CurrentYear] [YourBlogName.com]", "footer_align": "center",
            "title_page_enabled": False,
            "toc_enabled": False,
            "color_scheme": "Modern Gray", "include_tables_figures": True,
            "description": "A stylish template for engaging blog posts and online articles."
        },
        # Add more default templates for other PERSONA_CATEGORIES
        {
            "name": "General Purpose Document", "is_custom": False, "persona_category": "General/Others",
            "margin_top": 1.0, "margin_bottom": 1.0, "margin_left": 1.0, "margin_right": 1.0,
            "body_font": "Calibri", "body_font_size": 11,
            "heading_font": "Calibri Light", "heading1_font_size": 18, "heading1_bold": True,
            "heading2_font_size": 14, "heading2_bold": True,
            "heading3_font_size": 12, "heading3_bold": True,
            "line_spacing": 1.15, "paragraph_spacing_after": 8, "first_line_indent": 0.0,
            "header_text": "", "footer_text": "Page [PageNumber]", "footer_align": "right",
            "title_page_enabled": False, "toc_enabled": False,
            "color_scheme": "Default (Black/White)", "include_tables_figures": True,
            "description": "A clean, general-purpose template suitable for various document types."
        }
    ]
    # Add more template fields for advanced styling if needed (e.g., specific colors, heading spacing)
    for t in templates:
        t.setdefault("heading1_bold", True)
        t.setdefault("heading1_align", "left")
        t.setdefault("heading2_bold", True)
        t.setdefault("heading2_align", "left")
        t.setdefault("heading3_bold", False)
        t.setdefault("heading3_align", "left")
        t.setdefault("title_page_enabled", False)
        t.setdefault("toc_enabled", False)
    return templates


def initialize_default_formatting_templates(templates_collection, user_id_to_check="system_default"):
    """
    Initializes MongoDB with default formatting templates if they don't exist.
    These are system-wide defaults, not tied to a specific user initially,
    or can be assigned to a generic user_id like "system_default".
    Users can then select from these or create their own custom ones.
    """
    if templates_collection is None:
        logger.warning("Templates collection is None. Cannot initialize default formatting templates.")
        return

    default_templates_data = create_default_formatting_templates()
    # Check if default templates (marked by "is_custom": False) already exist for the system/generic user
    # A simpler check: if any default template name exists, assume they are all there.
    # A more robust check would verify each one or use a versioning system.
    
    # Count existing non-custom templates for the generic user_id
    existing_defaults_count = templates_collection.count_documents({"user_id": user_id_to_check, "is_custom": False})

    if existing_defaults_count >= len(default_templates_data):
        logger.info(f"Default formatting templates seem to be already initialized in the database for '{user_id_to_check}'.")
        return

    logger.info(f"Initializing default formatting templates in the database for '{user_id_to_check}'...")
    for template_data in default_templates_data:
        # Check if a template with this name for the system user already exists
        if not templates_collection.find_one({"name": template_data["name"], "user_id": user_id_to_check, "is_custom": False}):
            template_to_insert = template_data.copy()
            template_to_insert["template_id"] = str(uuid.uuid4()) # Unique ID for each template
            template_to_insert["user_id"] = user_id_to_check # Generic ID for system defaults
            template_to_insert["is_custom"] = False # Mark as a default, not user-created
            try:
                templates_collection.insert_one(template_to_insert)
                logger.info(f"Added default template: {template_data['name']}")
            except pymongo_errors.DuplicateKeyError:
                logger.warning(f"Default template '{template_data['name']}' likely already exists (duplicate key on unique index). Skipping.")
            except Exception as e:
                 logger.error(f"Failed to insert default template '{template_data['name']}': {e}")
        else:
            logger.info(f"Default template '{template_data['name']}' already exists for '{user_id_to_check}'. Skipping.")


def get_available_formatting_templates(templates_collection, current_user_id, user_tier):
    """
    Fetches available formatting templates: system defaults + user's custom templates.
    Filters system defaults based on user tier's allowed persona categories.
    """
    if templates_collection is None:
        logger.warning("MongoDB not available. Returning in-memory default templates filtered by tier.")
        all_defaults = create_default_formatting_templates()
        allowed_personas = UserTier.get_tier_features()[user_tier]["allowed_personas_for_llm"] # Using LLM personas as proxy
        
        # Filter default templates based on tier's allowed personas
        available_system_templates = [
            t for t in all_defaults if t.get("persona_category", "General/Others") in allowed_personas
        ]
        # Add temporary IDs for in-memory use
        for t in available_system_templates:
            t["template_id"] = str(uuid.uuid4())
            t["user_id"] = "system_default_in_memory"
            t["is_custom"] = False
        # For custom templates in memory mode, it would be empty unless session state handles it
        return available_system_templates

    # Fetch system default templates (user_id = "system_default", is_custom = False)
    system_default_templates = list(templates_collection.find({"user_id": "system_default", "is_custom": False}))

    # Filter system defaults based on the user's tier allowed persona categories
    allowed_personas = UserTier.get_tier_features()[user_tier]["allowed_personas_for_llm"]
    available_system_templates = [
        t for t in system_default_templates if t.get("persona_category", "General/Others") in allowed_personas
    ]
    
    # Fetch user's own custom templates
    user_custom_templates = list(templates_collection.find({"user_id": current_user_id, "is_custom": True}))
    
    return available_system_templates + user_custom_templates