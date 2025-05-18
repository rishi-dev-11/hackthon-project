import os
import logging
import fitz
import cv2
import numpy as np
from PIL import Image
import pdfplumber
import pandas as pd
from io import BytesIO
import tempfile

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

# === 1. Clean Column Names ===
def clean_column_names(df):
    """
    Ensure DataFrame column names are non-empty and unique.
    Replaces empty or duplicate columns with unique identifiers.
    """
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        col = str(col).strip()
        if not col:
            col = "Unnamed"
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
        new_cols.append(col)
    df.columns = new_cols
    return df

# === 2. Extract & Clean Tables from PDF ===
def extract_clean_tables_from_pdf(pdf_file):
    """
    Extract tables from a PDF and return a list of cleaned DataFrames.
    Args:
        pdf_file: Path to PDF file or file-like object.
    Returns:
        List of cleaned pandas DataFrames.
    """
    tables = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                if extracted:
                    for table_num, table in enumerate(extracted):
                        if not table or len(table) < 2:  # Skip empty tables or those without headers
                            continue
                        
                        # Create DataFrame with headers from first row
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = clean_column_names(df)
                        
                        # Add metadata
                        df.attrs['page_num'] = page_num + 1
                        df.attrs['table_num'] = table_num + 1
                        df.attrs['total_rows'] = len(df)
                        df.attrs['total_cols'] = len(df.columns)
                        
                        tables.append(df)
            
            logger.info(f"Extracted {len(tables)} tables from PDF")
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {str(e)}")
        
    return tables

# === 3. Convert Tables to Excel ===
def convert_tables_to_excel(tables, include_metadata=True):
    """
    Convert a list of DataFrames into an Excel binary file.
    Args:
        tables: List of pandas DataFrames.
        include_metadata: Whether to include table metadata in the Excel file.
    Returns:
        BytesIO object containing Excel file data.
    """
    if not tables:
        return None
    
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format for headers
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D9D9D9',
                'border': 1
            })
            
            for i, df in enumerate(tables):
                sheet_name = f'Table_{i+1}'
                
                # Write DataFrame to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                
                # Format headers
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 15)  # Set column width
                
                # Add metadata if requested
                if include_metadata and hasattr(df, 'attrs'):
                    metadata_row = len(df) + 3
                    worksheet.write(metadata_row, 0, "Metadata", workbook.add_format({'bold': True}))
                    worksheet.write(metadata_row + 1, 0, "Page Number")
                    worksheet.write(metadata_row + 1, 1, df.attrs.get('page_num', 'N/A'))
                    worksheet.write(metadata_row + 2, 0, "Table Number")
                    worksheet.write(metadata_row + 2, 1, df.attrs.get('table_num', 'N/A'))
                    worksheet.write(metadata_row + 3, 0, "Row Count")
                    worksheet.write(metadata_row + 3, 1, df.attrs.get('total_rows', len(df)))
                    worksheet.write(metadata_row + 4, 0, "Column Count")
                    worksheet.write(metadata_row + 4, 1, df.attrs.get('total_cols', len(df.columns)))
    except Exception as e:
        logger.error(f"Error converting tables to Excel: {str(e)}")
        return None
        
    output.seek(0)
    return output

# === 4. Extract Tables from PDF File ===
def extract_tables(pdf_path, output_format='excel', output_dir=None):
    """
    Extract tables from a PDF file and save them in the specified format.
    Args:
        pdf_path: Path to the PDF file.
        output_format: 'excel', 'csv', or 'json'.
        output_dir: Directory to save the extracted tables. If None, uses a temp dir.
    Returns:
        List of paths to the saved files.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract tables
    try:
        tables = extract_clean_tables_from_pdf(pdf_path)
        if not tables:
            logger.warning(f"No tables found in {pdf_path}")
            return []
            
        output_files = []
        
        # Save tables in the requested format
        if output_format == 'excel':
            # Save all tables to a single Excel file
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            excel_path = os.path.join(output_dir, f"{pdf_name}_tables.xlsx")
            
            excel_data = convert_tables_to_excel(tables)
            if excel_data:
                with open(excel_path, 'wb') as f:
                    f.write(excel_data.getvalue())
                output_files.append(excel_path)
        
        elif output_format == 'csv':
            # Save each table as a separate CSV file
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            for i, df in enumerate(tables):
                csv_path = os.path.join(output_dir, f"{pdf_name}_table_{i+1}.csv")
                df.to_csv(csv_path, index=False)
                output_files.append(csv_path)
                
        elif output_format == 'json':
            # Save each table as a separate JSON file
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            for i, df in enumerate(tables):
                json_path = os.path.join(output_dir, f"{pdf_name}_table_{i+1}.json")
                df.to_json(json_path, orient='records')
                output_files.append(json_path)
        
        return output_files
    
    except Exception as e:
        logger.error(f"Error processing PDF tables: {str(e)}")
        return []

# Default function for compatibility with existing code
def partition_pdf(*args, **kwargs):
    """Compatibility function for existing code references"""
    return []
    
# For Streamlit application
def run_table_editor_app(pdf_file=None):
    """
    Function to extract and edit tables from a PDF file.
    For use in a Streamlit application.
    """
    import streamlit as st
    
    st.set_page_config(page_title="PDF Table Editor", layout="wide")
    st.title("📄 PDF Table Extractor and Editor")

    uploaded_file = pdf_file or st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        tables = extract_clean_tables_from_pdf(uploaded_file)

        if tables:
            st.success(f"✅ Found {len(tables)} table(s). You can edit them below:")
            edited_tables = []

            for i, df in enumerate(tables):
                st.subheader(f"Table {i+1}")
                edited_df = st.data_editor(df, key=f"editor_{i}", num_rows="dynamic")
                edited_tables.append(edited_df)

            excel_data = convert_tables_to_excel(edited_tables)

            st.download_button(
                label="⬇ Download Edited Tables as Excel",
                data=excel_data,
                file_name="edited_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠ No tables were found in the uploaded PDF.")

# Run Streamlit app if executed directly
if __name__ == "__main__":
    run_table_editor_app()