# DocuMorph AI Fixes

This package contains fixes for several issues in the DocuMorph AI application:

1. **PyMuPDF Compatibility Fix**: Resolves `module 'fitz' has no attribute 'open'/'Document'` errors
2. **Table/Figure Extraction**: Improves extraction and rendering of tables and figures from PDFs
3. **Document Integration**: Ensures tables and figures appear properly in output documents
4. **Default Templates**: Adds templates for different user roles (Student, Researcher, etc.)
5. **Google Drive Integration**: Updates the redirect URI as specified

## Installation

### Automated Installation

1. Make sure the DocuMorph AI application is not currently running
2. Run the fix installer with:
   ```
   cd backend/version1.0.0
   python -m streamlit run integrate_fixes.py
   ```
3. Click the "Apply Fixes" button in the web UI
4. Restart the DocuMorph AI application

### Manual Installation

If the automated installation doesn't work, you can:

1. Copy `documorph_fixes.py` to the main backend directory
2. Update your imports in `documorph_ai.py`:
   ```python
   from documorph_fixes import (
       open_pdf, 
       extract_tables_from_pdf,
       extract_figures_from_pdf,
       add_table_to_docx,
       add_figure_to_docx
   )
   ```
3. Replace instances of `fitz.Document`, `FitzDocument`, etc. with `open_pdf`
4. Add template initialization in the main() function:
   ```python
   if docs_collection is not None and templates_collection is not None:
       from documorph_fixes import initialize_db_templates
       initialize_db_templates(st.session_state.user_id)
   ```

## Features

### PyMuPDF Compatibility

The fix provides a wrapper function that handles different versions of PyMuPDF, resolving the common "module has no attribute" errors by trying multiple approaches to open PDF files.

### Enhanced Table/Figure Handling

- Better extraction of tables with proper formatting
- Improved image extraction with validation and previews
- Base64 encoding of images for web preview
- Proper integration into DOCX documents

### Default Templates

Creates default document templates for different user roles:
- Student Essay
- Business Report
- Research Paper
- Blog Post
- Multilingual Document
- Book Chapter
- Team Document
- Project Report
- Custom Document

### Google Drive Integration

Updates the redirect URI to `http://localhost:8000/oauth2callback` as specified.

## Troubleshooting

If you encounter issues:

1. A backup of the original `documorph_ai.py` file is created as `documorph_ai_backup.py`
2. Check the console for detailed error messages
3. Install any missing dependencies:
   ```
   pip install pymupdf pandas pillow numpy
   ```

## Contact

If you have any questions or issues, please open a GitHub issue or contact the developer. 