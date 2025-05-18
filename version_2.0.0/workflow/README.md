# DocuMorph AI Workflow Tools

This directory contains workflow tools and documentation for the DocuMorph AI project.

## Contents

1. **project_techniques.html** - Visual diagram of techniques used in the project
2. **document_analyzer.py** - Streamlit app for analyzing document content (tables and figures)

## How to Use the Document Analyzer

The Document Analyzer is a Streamlit application that allows you to extract and analyze content from PDF documents.

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install streamlit pandas pdfplumber matplotlib numpy pymupdf Pillow
```

### Running the Application

Navigate to the workflow directory and run:

```bash
streamlit run document_analyzer.py
```

This will start the application and open it in your default web browser.

### Features

- **Table Extraction**: Extract tables from PDFs and export to Excel or CSV
- **Figure Extraction**: Extract images and figures from PDFs
- **Data Editing**: Edit extracted tables directly in the app
- **Visualization**: View content distribution across pages
- **Reporting**: Generate HTML reports of extracted content

## Project Techniques Diagram

The `project_techniques.html` file contains a visual representation of the architecture and techniques used in DocuMorph AI. Open it in a web browser to view the diagram.

Key components documented in the diagram:

- System Architecture (Frontend, Backend, Database)
- Technologies Used
- Document Processing Workflow
- Authentication Flow
- Data Extraction Techniques

## Integration with Main Application

The workflow tools can be integrated with the main DocuMorph AI application to enhance its capabilities:

1. The table extraction module can be used in the main backend to extract tables from uploaded documents
2. The figure extraction functionality can be integrated into the document processing pipeline
3. The visualization techniques can be added to the dashboard for better user insights

## Troubleshooting

- **ImportError**: Make sure you're running the application from the root directory or adjust the import paths in the script
- **Module not found**: Install any missing dependencies mentioned in the error message
- **Permission errors**: Ensure you have read/write permissions in the directory where the application is running 