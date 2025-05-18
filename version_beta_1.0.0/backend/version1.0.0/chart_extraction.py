import os
import logging
import fitz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Unstructured module
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import convert_to_dict
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured module not available.")
    def partition_pdf(*args, **kwargs):
        return []

def identify_and_extract_charts(page, page_num):
    """Extract text from charts on a page using unstructured."""
    chunks = []
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured unavailable for chart extraction.")
        return [f"Page {page_num+1} Chart: [Error: Unstructured unavailable]"]
    
    try:
        # Save page as a temporary PDF
        doc = fitz.Document()
        doc.insert_page(-1)
        doc[0].insert_image(doc[0].rect, pixmap=page.get_pixmap(matrix=fitz.Matrix(2, 2)))
        temp_pdf = f"temp_page_{page_num}_chart.pdf"
        doc.save(temp_pdf)
        doc.close()
        
        # Process with unstructured
        elements = partition_pdf(
            temp_pdf,
            extract_images=True,
            strategy="hi_res"
        )
        
        for element in elements:
            element_dict = convert_to_dict(element)
            if element_dict.get("type") in ["Image", "FigureCaption"]:
                chart_text = element_dict.get("metadata", {}).get("extracted_text", "") or element_dict.get("text", "")
                if chart_text.strip():
                    chunks.append(f"Page {page_num+1} Chart: {chart_text}")
        
        os.remove(temp_pdf)
        if not chunks:
            chunks.append(f"Page {page_num+1} Chart: [No charts detected]")
        logger.info(f"Extracted {len(chunks)} chart chunks from page {page_num+1}: {chunks}")
        return chunks
    except Exception as e:
        logger.error(f"Error in chart extraction: {e}", exc_info=True)
        return [f"Page {page_num+1} Chart: [Error: {str(e)}]"]