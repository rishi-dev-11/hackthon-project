import streamlit as st
import os
import logging
import tempfile
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for unstructured module
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.text import partition_text
    UNSTRUCTURED_AVAILABLE = True
    print("Unstructured module is available!")
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("Unstructured module is NOT available. Install with 'pip install unstructured unstructured-inference'")

def main():
    st.title("DocuMorph AI - Unstructured.io Test")
    
    st.write("This app tests the integration of unstructured.io for better document parsing.")
    
    if not UNSTRUCTURED_AVAILABLE:
        st.error("Unstructured module is not available. Please install it first.")
        st.code("pip install unstructured unstructured-inference")
        return
    
    uploaded_file = st.file_uploader("Upload a test document", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if st.button("Process with Unstructured.io"):
            try:
                st.info(f"Processing {uploaded_file.name} with unstructured.io...")
                
                if file_type == "pdf":
                    elements = partition_pdf(file_path, strategy="hi_res")
                elif file_type == "docx":
                    elements = partition_docx(file_path)
                elif file_type == "txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    elements = partition_text(text)
                
                st.success(f"Successfully processed document! Extracted {len(elements)} elements.")
                
                # Display element types and content
                st.subheader("Extracted Elements")
                
                for i, element in enumerate(elements[:20]):  # Show first 20 elements
                    with st.expander(f"Element {i+1}: {type(element).__name__}"):
                        if hasattr(element, "text"):
                            st.text(element.text[:1000] + "..." if len(element.text) > 1000 else element.text)
                        else:
                            st.text(f"Element has no text attribute: {element}")
                
                if len(elements) > 20:
                    st.info(f"Showing only the first 20 of {len(elements)} elements.")
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
                logger.exception("Error processing document")
        
        # Clean up
        os.unlink(file_path)

if __name__ == "__main__":
    main() 