import streamlit as st
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our extraction modules
try:
    from server.modules.table_extraction import extract_clean_tables_from_pdf, convert_tables_to_excel
    from server.modules.figure_extraction import extract_images_from_pdf, create_figure_report
except ImportError:
    st.error("Could not import extraction modules. Make sure you're running this from the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DocuMorph Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("DocuMorph Document Analyzer")
st.sidebar.markdown("Extract and analyze document components")

# Upload section
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a PDF file to extract tables and figures"
)

# Create tabs for different extraction functions
tab1, tab2, tab3 = st.tabs(["📋 Tables", "🖼️ Figures", "📊 Summary"])

# Initialize session state for storing results
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'figures' not in st.session_state:
    st.session_state.figures = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

# Process uploaded file
if uploaded_file is not None:
    # Save the file to temp directory
    temp_dir = st.session_state.temp_dir
    pdf_path = os.path.join(temp_dir, "uploaded_document.pdf")
    
    # Only process if file has changed
    if st.session_state.pdf_path != pdf_path or not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.session_state.pdf_path = pdf_path
        
        # Reset previous results
        st.session_state.tables = []
        st.session_state.figures = []
        
        with st.spinner("Processing document..."):
            # Extract tables
            st.session_state.tables = extract_clean_tables_from_pdf(pdf_path)
            
            # Extract figures
            figures_dir = os.path.join(temp_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            st.session_state.figures = extract_images_from_pdf(
                pdf_path, 
                output_dir=figures_dir,
                min_width=50,
                min_height=50,
                caption_detection=True
            )

    # Tables tab content
    with tab1:
        st.title("Table Extraction")
        
        if st.session_state.tables:
            st.success(f"Found {len(st.session_state.tables)} tables in the document")
            
            # Table options
            table_format = st.radio(
                "Export Format", 
                ["Excel (all tables)", "CSV (separate files)", "View in app"], 
                horizontal=True
            )
            
            # Display each table
            for i, df in enumerate(st.session_state.tables):
                with st.expander(f"Table {i+1} (Page {df.attrs.get('page_num', '?')})"):
                    st.dataframe(df, use_container_width=True)
                    
                    # Allow editing within the app
                    if table_format == "View in app":
                        st.session_state.tables[i] = st.data_editor(
                            df, 
                            key=f"table_editor_{i}",
                            num_rows="dynamic",
                            use_container_width=True
                        )
            
            # Download options
            if table_format == "Excel (all tables)":
                excel_data = convert_tables_to_excel(st.session_state.tables)
                if excel_data:
                    st.download_button(
                        "⬇️ Download All Tables (Excel)",
                        data=excel_data,
                        file_name="extracted_tables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            elif table_format == "CSV (separate files)":
                import zipfile
                import pandas as pd
                
                # Create ZIP with CSVs
                csv_zip_path = os.path.join(st.session_state.temp_dir, "tables_csv.zip")
                with zipfile.ZipFile(csv_zip_path, "w") as zipf:
                    for i, df in enumerate(st.session_state.tables):
                        csv_path = os.path.join(st.session_state.temp_dir, f"table_{i+1}.csv")
                        df.to_csv(csv_path, index=False)
                        zipf.write(csv_path, arcname=f"table_{i+1}.csv")
                
                # Download button
                with open(csv_zip_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download All Tables (CSV)",
                        data=f.read(),
                        file_name="tables_csv.zip",
                        mime="application/zip"
                    )
        else:
            st.info("No tables found in the document or document not yet processed")
            
    # Figures tab content
    with tab2:
        st.title("Figure Extraction")
        
        if st.session_state.figures:
            st.success(f"Found {len(st.session_state.figures)} figures in the document")
            
            # Figure display options
            col1, col2 = st.columns([3, 1])
            with col1:
                display_mode = st.radio(
                    "Display Mode", 
                    ["Grid", "Individual"], 
                    horizontal=True
                )
            with col2:
                create_report = st.checkbox("Generate HTML Report", value=True)
            
            # Display figures
            if display_mode == "Grid":
                # Create a grid layout
                cols = st.columns(3)
                for i, fig in enumerate(st.session_state.figures):
                    if fig.get("path") and os.path.exists(fig.get("path")):
                        with cols[i % 3]:
                            st.image(
                                fig.get("path"),
                                caption=f"Figure {i+1} (Page {fig.get('page_num', '?')})",
                                use_column_width=True
                            )
                            if fig.get("caption"):
                                st.caption(fig.get("caption"))
            else:
                # Display individual figures with more details
                for i, fig in enumerate(st.session_state.figures):
                    with st.expander(f"Figure {i+1} (Page {fig.get('page_num', '?')})"):
                        cols = st.columns([2, 1])
                        with cols[0]:
                            if fig.get("path") and os.path.exists(fig.get("path")):
                                st.image(fig.get("path"), use_column_width=True)
                                if fig.get("caption"):
                                    st.caption(fig.get("caption"))
                        with cols[1]:
                            st.json({
                                "page": fig.get("page_num"),
                                "dimensions": f"{fig.get('width')}×{fig.get('height')}",
                                "type": fig.get("figure_type"),
                                "caption": fig.get("caption") or "None detected"
                            })
            
            # Create and offer report download
            if create_report:
                report_dir = os.path.join(st.session_state.temp_dir, "report")
                os.makedirs(report_dir, exist_ok=True)
                report_path = create_figure_report(
                    st.session_state.figures,
                    output_dir=report_dir,
                    include_images=True,
                    format="html"
                )
                
                if report_path and os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_content = f.read()
                    
                    st.download_button(
                        "⬇️ Download HTML Report",
                        data=report_content,
                        file_name="figures_report.html",
                        mime="text/html"
                    )
            
            # Download all figures as ZIP
            import zipfile
            zip_path = os.path.join(st.session_state.temp_dir, "extracted_figures.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                # Add metadata JSON
                figures_dir = os.path.join(st.session_state.temp_dir, "figures")
                metadata_path = os.path.join(figures_dir, "figures_metadata.json")
                if os.path.exists(metadata_path):
                    zipf.write(metadata_path, arcname="figures_metadata.json")
                
                # Add figure files
                for fig in st.session_state.figures:
                    if fig.get("path") and os.path.exists(fig.get("path")):
                        zipf.write(fig.get("path"), arcname=fig.get("filename"))
            
            # Download button
            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Download All Figures (ZIP)",
                    data=f.read(),
                    file_name="extracted_figures.zip",
                    mime="application/zip"
                )
        else:
            st.info("No figures found in the document or document not yet processed")
    
    # Summary tab content
    with tab3:
        st.title("Document Summary")
        
        # Basic document info
        st.header("Document Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", "N/A")  # Would need PyMuPDF to get page count
        with col2:
            st.metric("Tables Extracted", len(st.session_state.tables))
        with col3:
            st.metric("Figures Extracted", len(st.session_state.figures))
        
        # Summary statistics
        st.header("Content Distribution")
        if st.session_state.tables or st.session_state.figures:
            # Create simple chart of tables and figures by page
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Collect page data
            page_data = {}
            
            for table in st.session_state.tables:
                page = table.attrs.get('page_num', 1)
                if page not in page_data:
                    page_data[page] = {'tables': 0, 'figures': 0}
                page_data[page]['tables'] += 1
                
            for figure in st.session_state.figures:
                page = figure.get('page_num', 1)
                if page not in page_data:
                    page_data[page] = {'tables': 0, 'figures': 0}
                page_data[page]['figures'] += 1
            
            # Convert to DataFrame for plotting
            df = pd.DataFrame.from_dict(page_data, orient='index').reset_index()
            df.columns = ['Page', 'Tables', 'Figures']
            df = df.sort_values('Page')
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x = np.arange(len(df))
            width = 0.35
            
            tables_bar = ax.bar(x, df['Tables'], width, label='Tables')
            figures_bar = ax.bar(x, df['Figures'], width, bottom=df['Tables'], label='Figures')
            
            ax.set_ylabel('Count')
            ax.set_xlabel('Page Number')
            ax.set_title('Tables and Figures by Page')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Page'])
            ax.legend()
            
            st.pyplot(fig)
            
            # Show content statistics in table
            st.subheader("Content Statistics")
            stats_df = pd.DataFrame({
                'Statistic': ['Total Tables', 'Total Figures', 'Pages with Content', 'Avg Tables per Page', 'Avg Figures per Page'],
                'Value': [
                    len(st.session_state.tables),
                    len(st.session_state.figures),
                    len(page_data),
                    round(len(st.session_state.tables) / max(len(page_data), 1), 2),
                    round(len(st.session_state.figures) / max(len(page_data), 1), 2)
                ]
            })
            st.table(stats_df)
            
        else:
            st.info("Process the document to see summary statistics")

else:
    # Show demo/intro when no file is uploaded
    st.title("DocuMorph Document Analyzer")
    st.markdown("""
    ## Extract and analyze tables and figures from documents
    
    Upload a PDF document to get started.
    
    ### Features:
    
    - **Table Extraction**: Identify and extract tables from documents
    - **Figure Extraction**: Detect and extract images and charts
    - **Data Editing**: Edit extracted tables directly in the app
    - **Export Options**: Download as Excel, CSV, or view online
    - **Document Summary**: Get statistics and visualization of document content
    
    ### How to use:
    
    1. Upload a PDF document using the sidebar
    2. Switch between tabs to view tables and figures
    3. Use the export options to download extracted content
    """)
    
    st.image("https://images.unsplash.com/photo-1532153975070-2e9ab71f1b14?q=80&w=1000", caption="Upload a document to get started")

# Run the app with: streamlit run document_analyzer.py 