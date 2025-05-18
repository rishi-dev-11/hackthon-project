import os
import warnings
import logging
from shutil import which
import re # Add re import
try: # Add BeautifulSoup import
    from bs4 import BeautifulSoup
    BS_AVAILABLE = True
except ImportError:
    BS_AVAILABLE = False
    logging.warning("BeautifulSoup not available. HTML table parsing might be less robust.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_with_unstructured(pdf_path):
    """Process a PDF document using unstructured.io."""
    try:
        # Check for Tesseract and Poppler
        if which('tesseract') is None:
            logger.error("Tesseract OCR not installed.")
            return ["Error: Tesseract OCR is not installed. Install Tesseract OCR and ensure it's in your PATH."]
        if which('pdftoppm') is None:
            logger.error("Poppler not installed.")
            return ["Error: Poppler is not installed. Install Poppler and ensure it's in your PATH."]
        
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.staging.base import convert_to_dict
        except ImportError:
            logger.error("Unstructured module not available.")
            return ["Error: Unstructured module not available. Install with 'pip install unstructured[pdf]'"]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elements = partition_pdf(
                pdf_path,
                extract_images=True,
                extract_tables=True,
                include_metadata=True,
                strategy="hi_res",
                infer_table_structure=True,
                chunking_strategy="by_title"
            )
        
        chunks = []
        for idx, element in enumerate(elements):
            element_dict = convert_to_dict(element)
            element_type = element_dict.get("type")
            if element_type in ["Text", "NarrativeText"]:
                chunks.append(f"[Text] {element_dict.get('text', '')}")
            elif element_type == "Title":
                chunks.append(f"[Title] {element_dict.get('text', '')}")
            elif element_type == "Table":
                formatted_table = ""
                html_table_data = element_dict.get("metadata", {}).get("text_as_html", "")
                
                if html_table_data and BS_AVAILABLE:
                    try:
                        soup = BeautifulSoup(html_table_data, "html.parser")
                        parsed_rows = []
                        for tr_tag in soup.find_all('tr'):
                            cols = [ele.get_text(separator=' ', strip=True) for ele in tr_tag.find_all(['th', 'td'])]
                            parsed_rows.append("|".join(cols))
                        formatted_table = "\n".join(parsed_rows)
                    except Exception as e:
                        logger.warning(f"BeautifulSoup parsing of HTML table failed: {e}. Falling back.")
                        formatted_table = "" # Reset on error to allow fallback

                if not formatted_table: # Fallback if HTML parsing failed or BS not available or no HTML
                    table_text_content = element_dict.get("text", "")
                    if table_text_content:
                        # Basic heuristic for plain text tables from Unstructured:
                        # Unstructured's element.text for tables is often somewhat structured.
                        # We'll primarily clean it up.
                        lines = [line.strip() for line in table_text_content.split('\n') if line.strip()]
                        cleaned_lines = []
                        for line in lines:
                            # Replace multiple spaces with a single pipe, then clean up common issues.
                            # This is a heuristic. Unstructured's `infer_table_structure` should do most heavy lifting.
                            line_piped = re.sub(r'\s{2,}', '|', line) # Convert multiple spaces to pipes
                            line_piped = re.sub(r'\s*\|\s*', '|', line_piped) # Normalize spaces around pipes
                            line_piped = line_piped.strip('|') # Remove leading/trailing pipes
                            cleaned_lines.append(line_piped)
                        formatted_table = "\n".join(cleaned_lines)
                
                if formatted_table:
                    chunks.append(f"[Table] {formatted_table}")
                    logger.info(f"Extracted table (source: {'HTML' if html_table_data and BS_AVAILABLE and formatted_table.startswith(parsed_rows[0] if 'parsed_rows' in locals() and parsed_rows else '') else 'text'}) from {pdf_path}: {formatted_table[:100]}...")

            elif element_type == "Image":
                img_text = element_dict.get("metadata", {}).get("extracted_text", "")
                if img_text:
                    chunks.append(f"[Image] {img_text}")
                    logger.info(f"Extracted image text from {pdf_path}: {img_text[:100]}...")
            elif element_type == "ListItem":
                chunks.append(f"[List Item] {element_dict.get('text', '')}")
            elif element_type == "Formula":
                chunks.append(f"[Formula] {element_dict.get('text', '')}")
            elif "text" in element_dict and element_dict.get('text', '').strip(): # Generic text catch-all
                chunks.append(f"[{element_type}] {element_dict.get('text', '')}")
        
        if not chunks:
            logger.warning("No content extracted from document.")
            return ["Error: No content could be extracted from the document."]
        logger.info(f"Processed {pdf_path} with {len(chunks)} chunks using Unstructured.io.")
        return chunks
    except Exception as e:
        logger.error(f"Error processing with unstructured.io: {e}", exc_info=True)
        return [f"Error processing document with unstructured.io: {str(e)}"]