# DocuMorph AI

DocuMorph AI is an intelligent document transformation platform that automates the process of formatting, structuring, and enhancing documents according to professional templates. It uses advanced AI techniques to analyze, enhance, and format unstructured documents into professionally styled outputs.

## Key Features

- **Document Upload & Processing**: 
  - Support for multiple file formats (DOCX, PDF, TXT, Images)
  - Intelligent document structure detection
  - Table and figure extraction
  - OCR for handwritten text using PaddleOCR

- **Template Management**: 
  - Role-based templates (Student, Researcher, Business Professional, etc.)
  - Customizable layout, typography, and styling
  - Header and footer configuration

- **AI-Powered Document Enhancement**:
  - Smart abstract generation
  - Section title suggestions
  - Style improvement and word choice enhancement
  - Document structure analysis and recommendations

- **Tables & Figures Management**:
  - Automatic detection and extraction
  - Intelligent caption generation
  - Proper numbering and referencing

- **Live Document Editor**:
  - Real-time editing with AI-powered word suggestions
  - Context-aware text completion

- **Export Options**:
  - DOCX document export
  - Google Drive integration for Premium users
  - Formatted preview before export

- **Additional Premium Features**:
  - Plagiarism checking
  - Multi-language support
  - Team collaboration
  - Style guide compliance

## Technical Stack

- **Frontend**: Streamlit for interactive UI
- **Backend**: Python with LangChain for document processing
- **AI/ML**:
  - Groq LLM for text generation and analysis
  - PaddleOCR for handwritten text recognition
  - HuggingFace embeddings for document similarity
- **Database**: MongoDB for document and template storage
- **Document Processing**:
  - PyMuPDF and python-docx for document manipulation
  - Langchain for document chunking and processing
  - FAISS for vector search capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- MongoDB (optional, for persistent storage)
- Tesseract OCR (for basic OCR functionality)
- PaddleOCR and PaddlePaddle (for enhanced handwritten text recognition)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/documorph-ai.git
   cd documorph-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Spacy language model:
   ```
   python -m spacy download en_core_web_sm
   ```

5. Run the application:
   ```
   cd backend
   streamlit run documorph_ai.py
   ```

## Usage Guide

1. **Document Upload**:
   - Select a document from your computer
   - Choose appropriate processing options
   - Wait for the AI to analyze the document structure

2. **Template Selection**:
   - Choose from pre-defined templates or create your own
   - Customize margins, fonts, colors, and styling

3. **Document Enhancement**:
   - Generate an abstract based on your content
   - Get intelligent section title suggestions
   - Enhance text style and word choice
   - Analyze and improve document structure

4. **Tables & Figures**:
   - Review automatically detected tables and figures
   - Configure numbering and captions
   - Position within the document

5. **Live Editing**:
   - Make final adjustments to your document
   - Get AI-powered word suggestions as you type

6. **Export**:
   - Preview the final document
   - Export to DOCX format
   - Premium users can export directly to Google Drive

## User Tiers

The application offers two tiers:

### Free Tier
- Basic document processing
- Limited templates
- Core formatting features
- Up to 5 documents

### Premium Tier
- Advanced AI-powered enhancements
- Multi-language support
- Plagiarism checking
- Google Drive integration
- Team collaboration features
- Up to 100 documents
- Advanced table and figure processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 