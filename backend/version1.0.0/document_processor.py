from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_document(pdf_path):
    """Process a PDF document using LangChain."""
    try:
        # Load PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Extract text from chunks
        text_chunks = [chunk.page_content for chunk in chunks]
        logger.info(f"Processed {pdf_path} with {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return [f"Error processing document: {str(e)}"]