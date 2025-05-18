import os
import logging
import fitz # PyMuPDF

# Configure logging
logger = logging.getLogger("DocuMorphAI")

# Check for Unstructured module
try:
    from unstructured.partition.pdf import partition_pdf as unstructured_partition_pdf_for_charts
    # from unstructured.staging.base import convert_to_dict # Not strictly needed if accessing element.text
    from unstructured.documents.elements import FigureCaption, Image as UnstructuredImageElement
    UNSTRUCTURED_FOR_CHART_TEXT_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_FOR_CHART_TEXT_AVAILABLE = False
    logger.warning("Unstructured module not available. Chart text extraction will be limited.")
    def unstructured_partition_pdf_for_charts(*args, **kwargs): return []

def identify_and_extract_chart_elements(page: fitz.Page, page_num: int, temp_dir: str):
    """
    Extracts text potentially related to charts/figures from a PDF page using unstructured.io.
    Returns a list of dictionaries, each representing a chart element with extracted text.
    """
    chart_elements_data = []
    if not UNSTRUCTURED_FOR_CHART_TEXT_AVAILABLE:
        logger.warning("Unstructured unavailable for chart text extraction.")
        return chart_elements_data

    temp_pdf_path_chart = None
    try:
        # Save the single page as a temporary PDF
        doc_temp = fitz.open()
        doc_temp.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
        temp_pdf_path_chart = os.path.join(temp_dir, f"temp_page_{page_num}_chart_text_input.pdf")
        doc_temp.save(temp_pdf_path_chart)
        doc_temp.close()

        elements = unstructured_partition_pdf_for_charts(
            filename=temp_pdf_path_chart,
            # strategy="hi_res", # Use hi_res to improve OCR if charts contain text
            # extract_images_in_pdf=True # This might extract the chart as an image file if needed
        )
        
        for el_idx, element in enumerate(elements):
            # We are looking for text around images or captions that might describe a chart
            if isinstance(element, (UnstructuredImageElement, FigureCaption)):
                chart_text = element.text.strip() if hasattr(element, 'text') else ""
                
                # Try to get image bytes if it's an image element and strategy involves extraction
                # This part is tricky as `partition_pdf` might not directly give image bytes
                # in the element object in all strategies.
                # Often, `extract_image_block_to_bytes` is needed or images are saved to disk.
                # For now, focus on text. If the whole page is rendered for `detect_figures_from_pdf`,
                # that can serve as the visual for "chart".

                if chart_text:
                    chart_elements_data.append({
                        "figure_id": f"chart_{page_num+1}_{el_idx}", # Treat as a type of figure
                        "page": page_num + 1,
                        "text_content": chart_text, # Text associated with/near the chart/image
                        "type": "chart_text_unstructured", # Indicate origin
                        "image_bytes": None, # We don't have isolated chart image bytes here
                        "preview_src": None, # No direct preview from this function
                         # Could add bounding_box from element.metadata if useful
                        "rect": getattr(element.metadata, 'coordinates', {}).get('points') if hasattr(element, 'metadata') else None
                    })
                    logger.info(f"Found potential chart/figure related text on page {page_num+1}: {chart_text[:100]}")

    except Exception as e:
        logger.error(f"Error in chart text extraction (page {page_num+1}): {e}", exc_info=True)
    finally:
        if temp_pdf_path_chart and os.path.exists(temp_pdf_path_chart):
            try:
                os.remove(temp_pdf_path_chart)
            except Exception as e_clean:
                logger.error(f"Error cleaning temp chart PDF {temp_pdf_path_chart}: {e_clean}")
    
    return chart_elements_data