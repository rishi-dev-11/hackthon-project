import os
import sys
import shutil
import logging
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our fixes
try:
    from documorph_fixes import (
        open_pdf, 
        extract_tables_from_pdf,
        extract_figures_from_pdf,
        add_table_to_docx,
        add_figure_to_docx,
        initialize_db_templates,
        setup_google_drive_auth
    )
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False
    logger.error("Could not import fixes. Make sure documorph_fixes.py is in the same directory.")
    
def apply_fixes_to_main_app():
    """Apply the fixes to the main DocuMorph AI application."""
    if not FIXES_AVAILABLE:
        st.error("Fix modules not available. Cannot apply fixes.")
        return False
        
    try:
        # Find the main application path
        backend_dir = Path(__file__).parent.parent
        main_app_path = backend_dir / "documorph_ai.py"
        
        if not main_app_path.exists():
            st.error(f"Main application not found at {main_app_path}")
            return False
            
        # Create a backup
        backup_path = backend_dir / "documorph_ai_backup.py"
        shutil.copy2(main_app_path, backup_path)
        logger.info(f"Created backup of main application at {backup_path}")
        
        # Read the main application code
        with open(main_app_path, 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        # Apply Fix 1: Replace PyMuPDF import and Document usage
        app_code = app_code.replace(
            'import fitz  # PyMuPDF\nfrom fitz import Document as FitzDocument  # Explicitly import Document', 
            'import fitz  # PyMuPDF\n# PyMuPDF compatibility wrapper\ndef open_pdf(pdf_path):\n    """Open a PDF with PyMuPDF using the most compatible method."""\n    try:\n        return fitz.open(pdf_path)\n    except AttributeError:\n        try:\n            return fitz.Document(pdf_path)\n        except AttributeError:\n            try:\n                from pymupdf import Document\n                return Document(pdf_path)\n            except (AttributeError, ImportError):\n                raise ImportError("Could not initialize PyMuPDF with any available method")'
        )
        
        # Replace all instances of FitzDocument with open_pdf
        app_code = app_code.replace('pdf_doc = FitzDocument(doc_path)', 'pdf_doc = open_pdf(doc_path)')
        app_code = app_code.replace('doc = FitzDocument(pdf_path)', 'doc = open_pdf(pdf_path)')
        app_code = app_code.replace('doc_pdf = FitzDocument(', 'doc_pdf = open_pdf(')
        
        # Apply Fix 2: Update table and figure extraction functions
        # Find the detect_tables_from_pdf function
        table_function_start = app_code.find('def detect_tables_from_pdf(')
        if table_function_start > 0:
            # Find the end of the function (next def statement)
            next_def = app_code.find('def ', table_function_start + 10)
            if next_def > 0:
                # Replace the function with our improved version
                old_function = app_code[table_function_start:next_def]
                new_function = '''def detect_tables_from_pdf(pdf_path):
    """Detect tables from a PDF document and return their content and location."""
    try:
        return extract_tables_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error detecting tables from PDF: {e}", exc_info=True)
        return []
'''
                app_code = app_code.replace(old_function, new_function)
        
        # Find the detect_figures_from_pdf function
        figure_function_start = app_code.find('def detect_figures_from_pdf(')
        if figure_function_start > 0:
            # Find the end of the function (next def statement)
            next_def = app_code.find('def ', figure_function_start + 10)
            if next_def > 0:
                # Replace the function with our improved version
                old_function = app_code[figure_function_start:next_def]
                new_function = '''def detect_figures_from_pdf(pdf_path):
    """Detect figures/images from a PDF document."""
    try:
        return extract_figures_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error detecting figures from PDF: {e}", exc_info=True)
        return []
'''
                app_code = app_code.replace(old_function, new_function)
        
        # Apply Fix 3: Update add_figure_to_docx and add_table_to_docx functions
        # Find the add_figure_to_docx function
        figure_add_start = app_code.find('def add_figure_to_docx(')
        if figure_add_start > 0:
            # Find the end of the function (next def statement)
            next_def = app_code.find('def ', figure_add_start + 10)
            if next_def > 0:
                # Replace the function with our improved version
                old_function = app_code[figure_add_start:next_def]
                new_function = '''def add_figure_to_docx(doc, figure, caption):
    """Add a figure to a DOCX document with proper numbering and caption."""
    from documorph_fixes import add_figure_to_docx as enhanced_add_figure
    return enhanced_add_figure(doc, figure, caption)
'''
                app_code = app_code.replace(old_function, new_function)
        
        # Find the add_table_to_docx function
        table_add_start = app_code.find('def add_table_to_docx(')
        if table_add_start > 0:
            # Find the end of the function (next def statement)
            next_def = app_code.find('def ', table_add_start + 10)
            if next_def > 0:
                # Replace the function with our improved version
                old_function = app_code[table_add_start:next_def]
                new_function = '''def add_table_to_docx(doc, table_data, caption):
    """Add a table to a DOCX document with proper numbering and caption."""
    from documorph_fixes import add_table_to_docx as enhanced_add_table
    return enhanced_add_table(doc, table_data, caption)
'''
                app_code = app_code.replace(old_function, new_function)
        
        # Apply Fix 4: Add template initialization code
        # Find the main() function
        main_start = app_code.find('def main():')
        if main_start > 0:
            # Find where session state is initialized
            session_state_init = app_code.find('if \'user_id\' not in st.session_state:', main_start)
            if session_state_init > 0:
                # Add template initialization after user_id is set
                template_init_code = '''
    # Initialize default templates if not already done
    if docs_collection is not None and templates_collection is not None and 'user_id' in st.session_state:
        from documorph_fixes import initialize_db_templates
        initialize_db_templates(st.session_state.user_id)
'''
                # Find end of session state initialization block
                end_of_block = app_code.find('\n    if', session_state_init + 10)
                if end_of_block > 0:
                    # Insert our code after the block
                    app_code = app_code[:end_of_block] + template_init_code + app_code[end_of_block:]
        
        # Apply Fix 5: Update Google Drive redirect URI
        export_gdrive_start = app_code.find('def export_to_google_drive(')
        if export_gdrive_start > 0:
            # Find the redirect_uri setting
            redirect_uri_line = app_code.find('redirect_uri=', export_gdrive_start)
            if redirect_uri_line > 0:
                # Find the end of the line
                end_line = app_code.find('\n', redirect_uri_line)
                if end_line > 0:
                    # Replace the line
                    old_line = app_code[redirect_uri_line:end_line]
                    new_line = 'redirect_uri=\'http://localhost:8000/oauth2callback\''
                    app_code = app_code.replace(old_line, new_line)
        
        # Apply Fix 6: Add imports for our fixes at the top of the file
        imports_end = 0
        for line in app_code.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                imports_end = app_code.find(line) + len(line)
        
        if imports_end > 0:
            fixes_import = '''

# Import fixes
try:
    from documorph_fixes import (
        open_pdf,
        extract_tables_from_pdf,
        extract_figures_from_pdf,
        add_table_to_docx as enhanced_add_table_to_docx,
        add_figure_to_docx as enhanced_add_figure_to_docx,
        initialize_db_templates,
        setup_google_drive_auth
    )
    FIXES_AVAILABLE = True
except ImportError:
    FIXES_AVAILABLE = False
    logging.warning("DocuMorph fixes not available. Some issues may persist.")
'''
            app_code = app_code[:imports_end] + fixes_import + app_code[imports_end:]
        
        # Write the modified code back to the file
        with open(main_app_path, 'w', encoding='utf-8') as f:
            f.write(app_code)
        
        logger.info("Successfully applied fixes to the main application")
        return True
    except Exception as e:
        logger.error(f"Error applying fixes: {e}", exc_info=True)
        # Try to restore backup if available
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, main_app_path)
                logger.info("Restored backup after error")
        except Exception as restore_error:
            logger.error(f"Could not restore backup: {restore_error}")
        return False

# Streamlit UI for applying fixes
def main():
    st.title("DocuMorph AI - Fix Installer")
    
    st.markdown("""
    ## About This Fix
    This tool will apply the following fixes to the DocuMorph AI application:
    
    1. **PyMuPDF Compatibility Fix**: Resolves the "module 'fitz' has no attribute..." errors
    2. **Table/Figure Extraction**: Improves extraction of tables and figures from PDFs
    3. **Document Integration**: Enhances how tables and figures are displayed in output documents
    4. **Default Templates**: Creates templates for different user roles
    5. **Google Drive Integration**: Updates the redirect URI to "http://localhost:8000/oauth2callback"
    """)
    
    if FIXES_AVAILABLE:
        st.success("Fix modules loaded successfully!")
    else:
        st.error("Fix modules not available. Make sure documorph_fixes.py is in the same directory.")
        st.stop()
    
    if st.button("Apply Fixes"):
        with st.spinner("Applying fixes..."):
            success = apply_fixes_to_main_app()
            
            if success:
                st.success("""
                ✅ Successfully applied all fixes to DocuMorph AI!
                
                You can now restart the application to use the new features.
                """)
            else:
                st.error("""
                ❌ Error applying fixes.
                
                Check the console for error details. A backup of the original file has been created if possible.
                """)

if __name__ == "__main__":
    main() 