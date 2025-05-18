import os
import logging
import fitz # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import re
import io # For BytesIO if parsing HTML tables

# Configure logging
logger = logging.getLogger("DocuMorphAI")

# Check for Unstructured module (primarily for partition_image if used for tables)
try:
    from unstructured.partition.image import partition_image as unstructured_partition_image_for_tables
    from unstructured.documents.elements import Table as UnstructuredTableElement
    UNSTRUCTURED_FOR_TABLE_IMAGE_METHOD = True
except ImportError:
    UNSTRUCTURED_FOR_TABLE_IMAGE_METHOD = False
    logger.warning("Unstructured (for image-based table extraction) module not available. Table extraction from images will be limited.")
    def unstructured_partition_image_for_tables(*args, **kwargs): return [] # Placeholder

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame column names are non-empty, string type, and unique.
    Replaces empty or duplicate columns with unique identifiers.
    (Adapted from your pdfplumber script - very useful utility)
    """
    if df is None or df.empty:
        return pd.DataFrame() # Return empty DataFrame if input is None or empty

    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for i, col_header in enumerate(cols):
        # Ensure column name is a string and stripped
        col_name = str(col_header).strip() if pd.notna(col_header) else f"Unnamed_Col_{i+1}"
        if not col_name: # If it was None or whitespace only
            col_name = f"Unnamed_Col_{i+1}"
        
        original_col_name = col_name 
        count = seen.get(original_col_name, 0)
        current_col_name_to_use = col_name
        if count > 0: 
            current_col_name_to_use = f"{original_col_name}_{count}"
        
        seen[original_col_name] = count + 1
        new_cols.append(current_col_name_to_use)
    
    df.columns = new_cols
    return df

def extract_tables_from_pdf_page_as_image(page: fitz.Page, page_num: int, temp_dir:str):
    """
    Extracts tables from a single PDF page by converting the page to an image
    and then using unstructured.io's image partitioner.
    Returns a list of dictionaries, each representing a table with its data.
    """
    extracted_tables_on_page = []
    if not UNSTRUCTURED_FOR_TABLE_IMAGE_METHOD:
        logger.warning("Unstructured for image-based table extraction not available.")
        return extracted_tables_on_page

    temp_img_path = None
    try:
        zoom = 3.0 # Good balance for OCR quality vs. processing time
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        temp_img_path = os.path.join(temp_dir, f"temp_page_{page_num}_table_img.png")
        pix.save(temp_img_path)

        elements = unstructured_partition_image_for_tables(
            filename=temp_img_path,
            # strategy="hi_res", # Can be slow, consider "auto" or "fast"
            infer_table_structure=True # Important for table parsing quality
        )

        for el_idx, element in enumerate(elements):
            if isinstance(element, UnstructuredTableElement): # Check specific type
                table_html = getattr(element.metadata, 'text_as_html', None)
                table_text = element.text
                df = None
                raw_data_list = []

                if table_html:
                    try:
                        dfs = pd.read_html(io.StringIO(table_html))
                        if dfs: df = dfs[0]
                    except ValueError: # Handle "No tables found" by pd.read_html
                        logger.debug(f"Pandas found no tables in HTML for element {el_idx} on page {page_num+1}. Trying text.")
                        if table_text: # Fallback to parsing plain text
                            raw_data_list = [re.split(r'\s{2,}|\|', row.strip()) for row in table_text.splitlines() if row.strip()]
                    except Exception as e_html:
                        logger.warning(f"Could not parse HTML table (page {page_num+1}): {e_html}. Trying text.")
                        if table_text:
                            raw_data_list = [re.split(r'\s{2,}|\|', row.strip()) for row in table_text.splitlines() if row.strip()]
                elif table_text:
                    raw_data_list = [re.split(r'\s{2,}|\|', row.strip()) for row in table_text.splitlines() if row.strip()]

                if not df and raw_data_list: # If df wasn't created from HTML, try from raw_data_list
                    try:
                        # Heuristic: if first row looks like a header (not all numeric or mostly non-empty strings)
                        if raw_data_list[0] and any(isinstance(cell, str) and cell.isalpha() for cell in raw_data_list[0]):
                            df = pd.DataFrame(raw_data_list[1:], columns=raw_data_list[0])
                        else:
                            df = pd.DataFrame(raw_data_list)
                    except Exception as e_df_raw:
                        logger.warning(f"Could not create DataFrame from raw text table (page {page_num+1}): {e_df_raw}")
                        df = pd.DataFrame(raw_data_list) # Attempt simple DF creation

                if df is not None and not df.empty:
                    df = clean_column_names(df.copy())
                    extracted_tables_on_page.append({
                        "table_id": f"img_uns_{page_num+1}_{el_idx}", "dataframe": df, "raw_data": df.values.tolist(), # Store list of lists for raw_data
                        "page": page_num + 1, "extraction_method": "unstructured_image"
                    })
                elif raw_data_list: # If DF creation failed but we have raw_data_list
                     extracted_tables_on_page.append({
                        "table_id": f"img_uns_raw_{page_num+1}_{el_idx}", "dataframe": None, "raw_data": raw_data_list,
                        "page": page_num + 1, "extraction_method": "unstructured_image_raw_text"
                    })


    except Exception as e:
        logger.error(f"Error in image-based table extraction for page {page_num+1}: {e}", exc_info=True)
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception as e_clean:
                logger.error(f"Error cleaning temp image file {temp_img_path}: {e_clean}")
    
    return extracted_tables_on_page