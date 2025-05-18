import os
import logging
import fitz
import cv2
import numpy as np
from PIL import Image

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

def extract_tables(page, page_num):
    """Extract tables from a PDF page using unstructured."""
    chunks = []
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured unavailable for table extraction.")
        return [f"Page {page_num+1} Table: [Error: Unstructured unavailable]"]
    
    try:
        # Save page as a high-resolution image
        zoom = 6  # Increased for better table detection
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        temp_img_path = f"temp_page_{page_num}.png"
        cv2.imwrite(temp_img_path, enhanced)
        
        # Save as temporary PDF
        doc = fitz.Document()
        doc.insert_page(-1)
        doc[0].insert_image(doc[0].rect, filename=temp_img_path)
        temp_pdf = f"temp_page_{page_num}.pdf"
        doc.save(temp_pdf)
        doc.close()
        os.remove(temp_img_path)
        
        # Process with unstructured
        elements = partition_pdf(
            temp_pdf,
            extract_tables=True,
            strategy="hi_res",
            infer_table_structure=True
        )
        
        for element in elements:
            element_dict = convert_to_dict(element)
            if element_dict.get("type") == "Table":
                table_data = element_dict.get("metadata", {}).get("text_as_html", "") or element_dict.get("text", "")
                # Clean and format table data as pipe-separated text
                table_text = table_data.replace('<table>', '').replace('</table>', '').replace('<tr>', '').replace('</tr>', '').replace('<td>', '').replace('</td>', '|').replace('<th>', '').replace('</th>', '|').strip()
                lines = [line.strip('|') for line in table_text.split('\n') if line.strip()]
                formatted_table = "\n".join(["|".join(cell.strip() for cell in line.split('|') if cell.strip()) for line in lines])
                if formatted_table:
                    chunks.append(f"Page {page_num+1} Table: {formatted_table}")
                    logger.info(f"Extracted table from page {page_num+1}: {formatted_table[:100]}...")
        
        # Fallback: Extract raw text if no tables detected
        if not chunks:
            elements = partition_pdf(temp_pdf, strategy="fast")
            raw_text = "\n".join([str(el) for el in elements if str(el).strip()])
            if raw_text:
                chunks.append(f"Page {page_num+1} Table: [Raw Text] {raw_text}")
                logger.info(f"Extracted raw text from page {page_num+1}: {raw_text[:100]}...")
            else:
                chunks.append(f"Page {page_num+1} Table: [No tables or text detected]")
        
        os.remove(temp_pdf)
        logger.info(f"Extracted {len(chunks)} table chunks from page {page_num+1}")
        return chunks
    except Exception as e:
        logger.error(f"Error in table extraction: {e}", exc_info=True)
        return [f"Page {page_num+1} Table: [Error: {str(e)}]"]