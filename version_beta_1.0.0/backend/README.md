# DocuMorph AI

DocuMorph AI is an intelligent document transformation platform that automates the process of formatting and structuring documents according to professional templates. It uses AI to analyze, enhance, and format unstructured documents into professionally styled outputs.

## Features

- **Document Upload & Processing**: Upload unformatted content in DOCX, PDF, or TXT formats for processing
- **Template Management**: Define or select formatting templates with customizable layout rules, font styles, margins, and more
- **LLM Document Enhancement**:
  - Abstract Generation: AI-generated abstracts based on document content
  - Section Title Suggestions: Intelligent title alternatives for document sections
  - Style Enhancement: Improve word choice and style according to target formats
  - Document Structure Suggestions: AI-powered analysis of logical document structure
- **Preview & Export**: Apply templates to documents and export the formatted result

## Technical Details

DocuMorph AI is built on the following technologies:
- Streamlit for the web interface
- LangChain for document processing and LLM integration
- Groq LLM for AI-powered text generation and document analysis
- MongoDB for document and template storage
- Python-DOCX for document manipulation and formatting
- FAISS for vector search capabilities

## Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure MongoDB is installed and running on your system

3. Run the application:
   ```
   streamlit run backend/documorph_ai.py
   ```

## Usage

1. **Upload a Document**:
   - Choose an unformatted document (DOCX, PDF, or TXT)
   - Process the document to extract its content and structure

2. **Create or Select a Template**:
   - Define formatting rules (margins, fonts, headers/footers)
   - Save templates for future use

3. **Enhance Document with AI**:
   - Generate an abstract
   - Get section title suggestions
   - Enhance text style
   - Analyze document structure

4. **Apply Template and Export**:
   - Preview the formatted document
   - Export to DOCX format

## Future Enhancements

- OCR integration for scanned documents
- Multi-language support
- Integration with Google Docs or MS Word
- Version control for formatted documents
- Advanced table and chart formatting 