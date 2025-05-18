import streamlit as st
import os
import uuid
import logging
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import pytesseract
import os 
import logging

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. Install with 'pip install pytesseract' for OCR fallback.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# Check for Unstructured module
try:
    from unstructur import process_with_unstructured
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured module not available.")
    def process_with_unstructured(path):
        return["unstructered not available"]

# Image text extraction with simplified preprocessing and Tesseract fallback
def extract_image_text(page, img_index):
    """Extract text from an image on a PDF page using unstructured with fallback to pytesseract."""
    if not UNSTRUCTURED_AVAILABLE and not PYTESSERACT_AVAILABLE:
        logger.warning("Neither unstructured nor pytesseract available for image text extraction.")
        return "No OCR tools available"
    try:
        image_list = page.get_images(full=True)
        if img_index >= len(image_list):
            logger.error(f"Image index {img_index} out of range for page {page.number + 1}.")
            return "Image index out of range"
        xref = image_list[img_index][0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        
        # Log image properties
        logger.info(f"Processing image {img_index} on page {page.number + 1}: size={image.size}, mode={image.mode}")
        
        # Preprocess image for OCR
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        temp_path = f"temp_image_{page.number}_{img_index}.png"
        cv2.imwrite(temp_path, thresh)
        
        # Try unstructured first
        image_text = ""
        if UNSTRUCTURED_AVAILABLE:
            try:
                from unstructured.partition.image import partition_image
                elements = partition_image(temp_path)
                image_text_parts = []
                for el in elements:
                    if hasattr(el, 'text') and el.text:
                        image_text_parts.append(el.text)
                    elif str(el).strip():
                        image_text_parts.append(str(el))
                image_text = "\n".join(image_text_parts) if image_text_parts else ""
                logger.info(f"Unstructured output for image {img_index} on page {page.number + 1}: {image_text[:100]}...")
            except Exception as e:
                logger.warning(f"Unstructured failed for image {img_index} on page {page.number + 1}: {e}")
        
        # Fallback to pytesseract if unstructured fails or returns no text
        if not image_text and PYTESSERACT_AVAILABLE:
            try:
                image_text = pytesseract.image_to_string(Image.open(temp_path), lang='eng')
                image_text = image_text.strip()
                logger.info(f"Pytesseract output for image {img_index} on page {page.number + 1}: {image_text[:100]}...")
            except Exception as e:
                logger.warning(f"Pytesseract failed for image {img_index} on page {page.number + 1}: {e}")
        
        os.remove(temp_path)
        
        if not image_text:
            logger.warning(f"No text extracted from image {img_index} on page {page.number + 1}")
            # Fallback: Extract raw text from page
            raw_text = page.get_text("text").strip()
            if raw_text:
                logger.info(f"Fallback: Extracted raw text from page {page.number + 1}: {raw_text[:100]}...")
                return f"[Raw Page Text] {raw_text}"
            return "No text detected in image"
        
        return image_text
    except Exception as e:
        logger.error(f"Error extracting text from image {img_index} on page {page.number + 1}: {e}", exc_info=True)
        # Fallback: Extract raw text from page
        raw_text = page.get_text("text").strip()
        if raw_text:
            logger.info(f"Fallback: Extracted raw text from page {page.number + 1}: {raw_text[:100]}...")
            return f"[Raw Page Text] {raw_text}"
        return f"Error extracting image text: {str(e)}"

# Simple document processor with LangChain
def simple_document_processor(pdf_path):
    """Process PDF using LangChain and extract image text."""
    chunks = []
    try:
        # Load and split text using LangChain
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_docs = text_splitter.split_documents(documents)
        chunks.extend([doc.page_content for doc in text_docs])
        
        # Extract image text
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, _ in enumerate(image_list):
                image_text = extract_image_text(page, img_index)
                if image_text and "Error" not in image_text and "No text detected" not in image_text:
                    chunks.append(f"Page {page_num + 1} Image {img_index}: {image_text}")
                else:
                    chunks.append(f"Page {page_num + 1} Image {img_index}: [Failed to extract text - {image_text}]")
        doc.close()
        logger.info(f"Processed {pdf_path} with {len(chunks)} chunks in Simple Mode.")
        return chunks
    except Exception as e:
        logger.error(f"Error in simple document processing: {e}", exc_info=True)
        return [f"Error processing document: {str(e)}"]

# Initialize MongoDB client
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["rag_db"]
        collection = db["documents"]
        client.server_info()
        logger.info("MongoDB connection established.")
        return collection
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}", exc_info=True)
        st.error(f"MongoDB connection error: {str(e)}")
        return None

# Initialize FAISS vector store with LangChain
@st.cache_resource
def init_vector_store(chat_id):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store_path = f"vector_stores/faiss_index_{chat_id}"
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            vector_store = FAISS.from_texts(["initialize"], embeddings)
            vector_store.save_local(vector_store_path)
        logger.info(f"Vector store initialized for chat {chat_id}.")
        return vector_store, vector_store_path
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}", exc_info=True)
        st.error(f"Error initializing vector store: {str(e)}")
        return None, None

# Initialize Groq LLM with LangChain
@st.cache_resource
def init_llm():
    try:
        return ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            api_key="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"
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}", exc_info=True)
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def semantic_search(query, vector_store, k=10):
    if vector_store is None:
        logger.warning("Vector store is None.")
        return []
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Semantic search completed for query: {query} with {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        st.error(f"Error during semantic search: {str(e)}")
        return []

def generate_response(query, contexts):
    llm = init_llm()
    if llm is None:
        logger.error("LLM initialization failed.")
        return "LLM initialization failed."
    try:
        context_texts = [doc.page_content for doc, _ in contexts]
        context_str = "\n\n".join(context_texts)
        logger.info(f"Context for query '{query}': {context_str[:200]}...")
        prompt = PromptTemplate(
            template="""You are an advanced document analysis assistant specializing in extracting and presenting information from complex documents, including tables, images, and text. The context may contain raw text, table data (e.g., rows separated by newlines, columns by pipes '|'), or image-extracted text. Tables are formatted in Markdown or pipe-separated format.

Context Information:
{context}

User Question: {question}

Instructions:
1. **Specific Value Queries** (e.g., "What is the value in row X, column Y?", "What course is Kiara taking?", "What was the Cost for Vegetables?"):
   - Search tables for the exact value. Tables can be in Markdown or pipe-separated format.
   - To answer, identify the row corresponding to the item mentioned in the question (e.g., 'Vegetables').
   - Then, identify the column corresponding to the requested attribute (e.g., 'Cost').
   - The value at the intersection of this row and column is the answer.
   - **Crucially, if the question asks for "Cost", you MUST look for a column explicitly named "Cost", "Total Cost", or similar. Do NOT use "Unit Price", "Price per item", or similar as a substitute for "Cost" if a dedicated "Cost" column exists. If both "Cost" and "Unit Price" (or similar terms) are present for an item, and the question is "What is the Cost for [item]?", you MUST use the value from the "Cost" column.**
   - Return ONLY the precise value with its formatting (e.g., '$100', 'Soft Skills').
   - Example 1: For "What was the Cost for Vegetables?" with context "Item|Unit Price|Cost\nVegetables|$10|$100", the answer is: $100
   - Example 2: For "What course is Kiara taking?" with context "Name|Course\nKiara|Soft Skills\nJia|Cloud Computing", the answer is: Soft Skills

2. **Table Generation Queries** (e.g., "Generate a table", "Create a table of names and courses"):
   - Identify all relevant table data (e.g., 'Name|Course\nKiara|Soft Skills\nJia|Cloud Computing').
   - Extract or infer headers from the context or question.
   - Format as a clean Markdown table.
   - If headers are missing, infer from the question (e.g., 'names and courses' implies 'Name|Course').
   - Return ONLY the Markdown table.
   - Example: For 'Name|Course\nKiara|Soft Skills\nJia|Cloud Computing', return:
     ```
     | Name  | Course         |
     |-------|----------------|
     | Kiara | Soft Skills    |
     | Jia   | Cloud Computing|
     ```

3. **General Information Queries**:
   - Extract concise answers from tables, images, or text.
   - Prioritize table data for structured queries, then image text, then regular text.
   - Return a brief, precise answer.

4. **Chart/Graph Queries**:
   - Extract specific data points from chart descriptions or image text.
   - Maintain formatting and reference the chart element.

5. **If Information Is Missing**:
   - Reply with "The requested information is not found in the provided document."
   - Do not invent or extrapolate data.

Answer:""",
            input_variables=["context", "question"],
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context_str, "question": query})
        logger.info(f"Generated response for query '{query}': {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return f"Error generating response: {str(e)}"

def main():
    st.title("Multimodal RAG System for Complex Documents")

    # Initialize session state for chat management
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    if 'chat_ids' not in st.session_state:
        st.session_state.chat_ids = [st.session_state.chat_id]

    # Sidebar for system status and chat management
    with st.sidebar:
        st.header("System Status")
        pymupdf_status = "✅ Available" if 'fitz' in globals() else "❌ Not Available"
        unstructured_status = "✅ Available" if UNSTRUCTURED_AVAILABLE else "❌ Not Available"
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
            client.server_info()
            mongodb_status = "✅ Connected"
        except Exception:
            mongodb_status = "❌ Not Connected"

        st.markdown("### Core Dependencies")
        st.markdown(f"- **PyMuPDF**: {pymupdf_status}")
        st.markdown(f"- **Unstructured**: {unstructured_status}")
        st.markdown(f"- **MongoDB**: {mongodb_status}")

        # Chat management
        st.header("Chat Management")
        if st.button("New Chat"):
            new_chat_id = str(uuid.uuid4())
            st.session_state.chat_ids.append(new_chat_id)
            st.session_state.chat_id = new_chat_id
            logger.info(f"Created new chat: {new_chat_id}")
        selected_chat = st.selectbox("Select Chat", st.session_state.chat_ids, index=st.session_state.chat_ids.index(st.session_state.chat_id))
        if selected_chat != st.session_state.chat_id:
            st.session_state.chat_id = selected_chat
            logger.info(f"Switched to chat: {selected_chat}")

        # Processing mode
        st.header("Processing Mode")
        processing_mode = st.radio(
            "Select document processing mode:",
            ["Simple Mode", "Advanced Mode (Unstructured)"]
        )

        # Document upload
        st.header("Document Processing")
        uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

        if uploaded_file is not None and st.button("Process Document"):
            collection = init_mongodb()
            if collection is None:
                st.error("MongoDB connection failed.")
                logger.error("MongoDB connection not established.")
                return
            with st.spinner("Processing document..."):
                # Save PDF permanently
                chat_id = st.session_state.chat_id
                pdf_dir = f"pdfs/{chat_id}"
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Process document
                if processing_mode == "Simple Mode":
                    all_chunks = simple_document_processor(pdf_path)
                    logger.info(f"Processed {uploaded_file.name} in Simple Mode")
                else:  # Advanced Mode (Unstructured)
                    if UNSTRUCTURED_AVAILABLE:
                        all_chunks = process_with_unstructured(pdf_path)
                        if all_chunks and not all_chunks[0].startswith("Error"):
                            logger.info(f"Processed {uploaded_file.name} with Unstructured.io")
                        else:
                            all_chunks = simple_document_processor(pdf_path)
                            logger.warning(f"Unstructured failed for {uploaded_file.name}, used Simple Mode")
                    else:
                        all_chunks = simple_document_processor(pdf_path)
                        logger.warning(f"Unstructured not available for {uploaded_file.name}, used Simple Mode")

                if not all_chunks:
                    st.error("No text extracted from the document.")
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                    logger.error(f"No text extracted from {uploaded_file.name}.")
                    return

                # Store metadata in MongoDB
                vector_store, vector_store_path = init_vector_store(chat_id)
                if vector_store is None:
                    st.error("Failed to initialize vector store.")
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                    logger.error(f"Vector store initialization failed for {uploaded_file.name}.")
                    return
                try:
                    doc_id = collection.insert_one({
                        "chat_id": chat_id,
                        "title": uploaded_file.name,
                        "pdf_path": pdf_path,
                        "vector_path": vector_store_path
                    }).inserted_id
                    logger.info(f"Stored metadata for {uploaded_file.name} in MongoDB with doc_id: {doc_id}")
                except Exception as e:
                    st.error(f"Error storing document metadata: {str(e)}")
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                    logger.error(f"Error storing metadata for {uploaded_file.name}: {e}", exc_info=True)
                    return

                # Update vector store
                try:
                    new_vector_store = FAISS.from_texts(
                        all_chunks, 
                        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    )
                    vector_store.merge_from(new_vector_store)
                    vector_store.save_local(vector_store_path)
                    st.success(f"Processed and indexed {uploaded_file.name}")
                    logger.info(f"Indexed {uploaded_file.name} in chat {chat_id} with {len(all_chunks)} chunks")
                except Exception as e:
                    st.error(f"Error indexing document: {str(e)}")
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                    logger.error(f"Error indexing {uploaded_file.name}: {e}", exc_info=True)

    # Query interface
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Ask Questions About Your Documents")
        query = st.text_input("Enter your question:")
        if query and st.button("Search"):
            try:
                with st.spinner("Searching..."):
                    vector_store, _ = init_vector_store(st.session_state.chat_id)
                    search_results = semantic_search(query, vector_store)
                    if search_results:
                        response = generate_response(query, search_results)
                        st.subheader("Answer:")
                        st.write(response)
                    else:
                        st.warning("No relevant information found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Error during query: {e}", exc_info=True)

    with col2:
        if query and 'search_results' in locals() and search_results:
            st.header("Relevant Contexts")
            for i, (doc, score) in enumerate(search_results):
                with st.expander(f"Context {i+1} - Score: {score:.4f}"):
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()