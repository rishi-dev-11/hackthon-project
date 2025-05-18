import streamlit as st
import os
import logging
import uuid
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import UserTier # Import UserTier from config.py

logger = logging.getLogger(__name__)

@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        db = client["documorph_db"]
        docs_collection = db["documents"]
        templates_collection = db["templates"]
        users_collection = db["users"]
        client.server_info()
        logger.info("MongoDB connection established.")
        return docs_collection, templates_collection, users_collection
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}", exc_info=True)
        st.warning("MongoDB not available. Some features will be limited. Running in memory-only mode.")
        return None, None, None

@st.cache_resource
def init_vector_store(user_id):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store_path = f"vector_stores/faiss_index_{user_id}"
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True) # Ensure directory exists
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            # Create a dummy document to initialize FAISS if it's new
            vector_store = FAISS.from_texts(["initialize"], embeddings)
            vector_store.save_local(vector_store_path)
        logger.info(f"Vector store initialized for user {user_id}.")
        return vector_store, vector_store_path
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}", exc_info=True)
        st.error(f"Error initializing vector store: {str(e)}")
        return None, None

def create_default_templates():
    """Create default templates for different roles."""
    templates = [
        {
            "name": "Student Essay", "role": "Student", "margin_top": 1.0, "margin_bottom": 1.0, 
            "margin_left": 1.0, "margin_right": 1.0, "body_font": "Times New Roman", "body_font_size": 12, 
            "heading_font": "Times New Roman", "heading1_font_size": 16, "heading2_font_size": 14, 
            "heading3_font_size": 12, "line_spacing": 2.0, "paragraph_spacing": 6, 
            "header_text": "", "footer_text": "Page [Page] of [Pages]", 
            "include_tables_figures": True, "color_scheme": "Default (Black/White)"
        },
        {
            "name": "Business Report", "role": "Business Professional", "margin_top": 1.0, "margin_bottom": 1.0, 
            "margin_left": 1.25, "margin_right": 1.25, "body_font": "Calibri", "body_font_size": 11, 
            "heading_font": "Calibri", "heading1_font_size": 16, "heading2_font_size": 14, 
            "heading3_font_size": 12, "line_spacing": 1.15, "paragraph_spacing": 6, 
            "header_text": "[Company Name]", "footer_text": "Confidential | [Date] | Page [Page]", 
            "include_tables_figures": True, "color_scheme": "Blue Professional"
        },
        {
            "name": "Research Paper", "role": "Researcher", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.0, "margin_right": 1.0, "body_font": "Times New Roman", "body_font_size": 12,
            "heading_font": "Times New Roman", "heading1_font_size": 14, "heading2_font_size": 12,
            "heading3_font_size": 12, "line_spacing": 1.5, "paragraph_spacing": 6,
            "header_text": "", "footer_text": "[Page]",
            "include_tables_figures": True, "color_scheme": "Academic"
        },
        {
            "name": "Blog Post", "role": "Content Creator", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.0, "margin_right": 1.0, "body_font": "Arial", "body_font_size": 11,
            "heading_font": "Georgia", "heading1_font_size": 18, "heading2_font_size": 16,
            "heading3_font_size": 14, "line_spacing": 1.5, "paragraph_spacing": 12,
            "header_text": "", "footer_text": "© [Year] [Author Name]",
            "include_tables_figures": True, "color_scheme": "Modern Gray"
        },
         {
            "name": "Multilingual Document", "role": "Multilingual User", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.25, "margin_right": 1.25, "body_font": "Arial", "body_font_size": 11,
            "heading_font": "Arial", "heading1_font_size": 16, "heading2_font_size": 14,
            "heading3_font_size": 12, "line_spacing": 1.5, "paragraph_spacing": 6,
            "header_text": "", "footer_text": "[Page]/[Pages]",
            "include_tables_figures": True, "color_scheme": "Default (Black/White)"
        },
        {
            "name": "Book Chapter", "role": "Author", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.25, "margin_right": 1.25, "body_font": "Georgia", "body_font_size": 12,
            "heading_font": "Georgia", "heading1_font_size": 18, "heading2_font_size": 16,
            "heading3_font_size": 14, "line_spacing": 1.5, "paragraph_spacing": 6,
            "header_text": "[Book Title] - Chapter [Chapter]", "footer_text": "[Author Name] | Page [Page]",
            "include_tables_figures": True, "color_scheme": "Earthy Tones"
        },
        {
            "name": "Team Document", "role": "Collaborator", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.0, "margin_right": 1.0, "body_font": "Calibri", "body_font_size": 11,
            "heading_font": "Calibri", "heading1_font_size": 16, "heading2_font_size": 14,
            "heading3_font_size": 12, "line_spacing": 1.15, "paragraph_spacing": 6,
            "header_text": "[Team Name] - [Project Name]", "footer_text": "Last updated: [Date] | Page [Page]",
            "include_tables_figures": True, "color_scheme": "Blue Professional"
        },
        {
            "name": "Project Report", "role": "Project Manager", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.25, "margin_right": 1.25, "body_font": "Arial", "body_font_size": 11,
            "heading_font": "Arial", "heading1_font_size": 16, "heading2_font_size": 14,
            "heading3_font_size": 12, "line_spacing": 1.15, "paragraph_spacing": 6,
            "header_text": "[Project Name] - [Status]", "footer_text": "Confidential | Page [Page] of [Pages]",
            "include_tables_figures": True, "color_scheme": "Modern Gray"
        },
        {
            "name": "Custom Document", "role": "Others", "margin_top": 1.0, "margin_bottom": 1.0,
            "margin_left": 1.0, "margin_right": 1.0, "body_font": "Calibri", "body_font_size": 11,
            "heading_font": "Calibri", "heading1_font_size": 16, "heading2_font_size": 14,
            "heading3_font_size": 12, "line_spacing": 1.15, "paragraph_spacing": 6,
            "header_text": "", "footer_text": "Page [Page]",
            "include_tables_figures": True, "color_scheme": "Default (Black/White)"
        }
    ]
    return templates

# This function is imported from documorph_fixes in the original code.
# If documorph_fixes.initialize_db_templates is preferred, this can be a wrapper or removed.
# For now, I'm providing the original logic here as a fallback or alternative.
def initialize_db_templates_internal(user_id, templates_collection_obj):
    """Initialize MongoDB with default templates for a user if templates_collection_obj is provided."""
    if templates_collection_obj is None:
        logger.warning("Templates collection is None, cannot initialize DB templates.")
        return False
    try:
        existing_templates = list(templates_collection_obj.find({"user_id": user_id}))
        if existing_templates:
            logger.info(f"Templates already exist for user {user_id}")
            return True
            
        templates = create_default_templates()
        for template in templates:
            template_doc = template.copy() # Avoid modifying the original list dicts
            template_doc["user_id"] = user_id
            template_doc["template_id"] = str(uuid.uuid4())
            template_doc["is_custom"] = False # Default templates are not custom
            templates_collection_obj.insert_one(template_doc)
            
        logger.info(f"Initialized default templates for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error initializing DB templates: {e}", exc_info=True)
        return False

def get_templates_for_user(user_id, templates_collection_obj, user_tier=UserTier.FREE):
    if templates_collection_obj is None:
        logger.warning("Templates collection is None, returning in-memory default templates.")
        templates = create_default_templates()
        # Filter templates based on user tier if needed, or simply return all for in-memory
        # For simplicity, returning all, assuming UI will filter if necessary
        return [dict(t, user_id=user_id, template_id=str(uuid.uuid4()), is_custom=False) for t in templates]
    else:
        return list(templates_collection_obj.find({"user_id": user_id}))