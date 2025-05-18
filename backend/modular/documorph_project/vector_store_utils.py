import streamlit as st
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import VECTOR_STORE_DIR # Import from config

logger = logging.getLogger(__name__)

@st.cache_resource
def init_vector_store(user_id):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Use VECTOR_STORE_DIR from config
        vector_store_path = os.path.join(VECTOR_STORE_DIR, f"faiss_index_{user_id}")
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True) # Ensure directory exists

        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            # Create a dummy document to initialize if the store is new
            vector_store = FAISS.from_texts(["initialize vector store"], embeddings)
            vector_store.save_local(vector_store_path)
        logger.info(f"Vector store initialized for user {user_id} at {vector_store_path}.")
        return vector_store, vector_store_path
    except Exception as e:
        logger.error(f"Error initializing vector store for user {user_id}: {e}", exc_info=True)
        st.error(f"Error initializing vector store: {str(e)}")
        return None, None