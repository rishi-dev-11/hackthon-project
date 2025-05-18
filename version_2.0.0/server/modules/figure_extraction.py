import os
import logging
import tempfile
from PIL import Image
import io
import fitz  # PyMuPDF
import numpy as np
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FigureMetadata:
    """Class to store metadata about extracted figures"""
    def __init__(self, figure_id, page_num, bbox, caption=None, figure_type=None):
        self.figure_id = figure_id
        self.page_num = page_num
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.caption = caption
        self.figure_type = figure_type or "image"
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
        
    def to_dict(self):
        """Convert metadata to dictionary"""
        return {
            "figure_id": self.figure_id,
            "page_num": self.page_num,
            "bbox": self.bbox,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "width": self.width,
            "height": self.height,
            "area": self.area
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create metadata from dictionary"""
        return cls(
            figure_id=data.get("figure_id"),
            page_num=data.get("page_num"),
            bbox=data.get("bbox"),
            caption=data.get("caption"),
            figure_type=data.get("figure_type")
        )

def extract_images_from_pdf(pdf_path, output_dir=None, min_width=100, min_height=100, caption_detection=False):
    """
    Extract images from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images (created if it doesn't exist)
        min_width: Minimum width for extracted images
        min_height: Minimum height for extracted images
        caption_detection: Whether to attempt to detect captions below figures
    
    Returns:
        List of dictionaries with image metadata
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []
        
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # List to store image metadata
    extracted_images = []
    
    try:
        document = fitz.open(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(document):
            # Get images
            image_list = page.get_images(full=True)
            
            # Process each image
            for img_index, img_info in enumerate(image_list):
                try:
                    # Get image data
                    xref = img_info[0]
                    base_image = document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image for processing
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # Skip small images (likely icons or decorative elements)
                    if width < min_width or height < min_height:
                        continue
                        
                    # Get image rectangle (position on the page)
                    # This is approximate as we don't have direct rectangle info
                    # for images extracted with extract_image
                    rect = None
                    for img_rect in page.get_image_rects():
                        if img_rect[0] == xref:
                            rect = img_rect[1]
                            break
                    
                    # Generate unique ID and filename
                    figure_id = str(uuid.uuid4())
                    output_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save the image
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Create metadata
                    bbox = rect.irect if rect else (0, 0, width, height)
                    metadata = FigureMetadata(
                        figure_id=figure_id,
                        page_num=page_num + 1,
                        bbox=bbox,
                        figure_type=image_ext.upper(),
                    )
                    
                    # Look for potential caption if requested
                    if caption_detection and rect:
                        # Look for text below the image (common caption location)
                        caption_area = fitz.Rect(
                            rect.x0, rect.y1, rect.x1, rect.y1 + 50
                        )
                        caption_text = page.get_text("text", clip=caption_area).strip()
                        
                        # If text starts with "Figure" or "Fig", it's likely a caption
                        if caption_text and (caption_text.startswith("Figure") or 
                                           caption_text.startswith("Fig") or
                                           caption_text.startswith("fig") or
                                           caption_text.startswith("FIGURE")):
                            metadata.caption = caption_text
                    
                    # Store the result
                    result = metadata.to_dict()
                    result["filename"] = output_filename
                    result["path"] = output_path
                    extracted_images.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_index} on page {page_num+1}: {str(e)}")
        
        # Save metadata file
        metadata_path = os.path.join(output_dir, "figures_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(extracted_images, f, indent=2)
            
        logger.info(f"Extracted {len(extracted_images)} images from {pdf_path}")
        return extracted_images
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
        return []

def create_figure_report(figures, output_dir, include_images=True, format="html"):
    """
    Create a report of extracted figures
    
    Args:
        figures: List of figure metadata dictionaries
        output_dir: Directory to save the report
        include_images: Whether to include image files in the report
        format: Output format ("html" or "md")
        
    Returns:
        Path to the report file
    """
    if not figures:
        logger.warning("No figures to include in report")
        return None
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Sort figures by page number
    figures = sorted(figures, key=lambda x: (x.get("page_num", 0), x.get("figure_id", "")))
    
    report_path = os.path.join(output_dir, f"figures_report.{format}")
    
    try:
        if format == "html":
            # Create HTML report
            with open(report_path, "w") as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extracted Figures Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .figure-container { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .figure-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
        .figure-image { max-width: 100%; margin-top: 10px; }
        .metadata { font-size: 14px; color: #555; }
        h1, h2 { color: #333; }
        .caption { font-style: italic; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Extracted Figures Report</h1>
    <p>Total figures: {}</p>
""".format(len(figures)))
                
                # Add each figure
                for i, fig in enumerate(figures):
                    f.write(f"""
    <div class="figure-container">
        <div class="figure-header">
            <h2>Figure {i+1}</h2>
            <div class="metadata">Page {fig.get('page_num', 'Unknown')}</div>
        </div>
""")
                    # Add caption if available
                    if fig.get("caption"):
                        f.write(f'        <div class="caption">{fig.get("caption")}</div>\n')
                        
                    # Add image if requested and path exists
                    if include_images and fig.get("path") and os.path.exists(fig.get("path")):
                        # Use relative path for the image
                        rel_path = os.path.relpath(fig.get("path"), output_dir)
                        f.write(f'        <img class="figure-image" src="{rel_path}" alt="Figure {i+1}" />\n')
                    
                    # Add metadata
                    f.write(f"""
        <div class="metadata">
            <p>ID: {fig.get('figure_id', 'Unknown')}</p>
            <p>Type: {fig.get('figure_type', 'Unknown')}</p>
            <p>Dimensions: {fig.get('width', '?')} x {fig.get('height', '?')} pixels</p>
        </div>
    </div>
""")
                
                f.write("""
</body>
</html>
""")
                
        elif format == "md":
            # Create Markdown report
            with open(report_path, "w") as f:
                f.write("# Extracted Figures Report\n\n")
                f.write(f"Total figures: {len(figures)}\n\n")
                
                for i, fig in enumerate(figures):
                    f.write(f"## Figure {i+1}\n\n")
                    f.write(f"- **Page**: {fig.get('page_num', 'Unknown')}\n")
                    f.write(f"- **ID**: {fig.get('figure_id', 'Unknown')}\n")
                    f.write(f"- **Type**: {fig.get('figure_type', 'Unknown')}\n")
                    f.write(f"- **Dimensions**: {fig.get('width', '?')} x {fig.get('height', '?')} pixels\n")
                    
                    if fig.get("caption"):
                        f.write(f"\n*{fig.get('caption')}*\n\n")
                        
                    if include_images and fig.get("path") and os.path.exists(fig.get("path")):
                        rel_path = os.path.relpath(fig.get("path"), output_dir)
                        f.write(f"\n![Figure {i+1}]({rel_path})\n\n")
                    
                    f.write("\n---\n\n")
        
        logger.info(f"Created figure report at {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error creating figure report: {str(e)}")
        return None

def run_figure_extractor_app(pdf_file=None):
    """
    Function to extract figures from a PDF file.
    For use in a Streamlit application.
    """
    import streamlit as st
    
    st.set_page_config(page_title="PDF Figure Extractor", layout="wide")
    st.title("📊 PDF Figure Extractor")
    
    uploaded_file = pdf_file or st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Create temp directory for extraction
        output_dir = tempfile.mkdtemp()
        
        # Save uploaded file to temp location
        temp_pdf_path = os.path.join(output_dir, "uploaded.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Show extraction options
        st.subheader("Extraction Options")
        col1, col2 = st.columns(2)
        with col1:
            min_width = st.number_input("Minimum Figure Width (px)", value=100, min_value=10)
            caption_detection = st.checkbox("Attempt Caption Detection", value=True)
        with col2:
            min_height = st.number_input("Minimum Figure Height (px)", value=100, min_value=10)
        
        # Extract button
        if st.button("Extract Figures"):
            with st.spinner("Extracting figures from PDF..."):
                figures = extract_images_from_pdf(
                    temp_pdf_path, 
                    output_dir=output_dir,
                    min_width=min_width,
                    min_height=min_height,
                    caption_detection=caption_detection
                )
                
                if figures:
                    st.success(f"✅ Extracted {len(figures)} figures from the PDF")
                    
                    # Display figures
                    for i, fig in enumerate(figures):
                        with st.expander(f"Figure {i+1} (Page {fig.get('page_num', '?')})"):
                            cols = st.columns([2, 1])
                            with cols[0]:
                                if fig.get("path") and os.path.exists(fig.get("path")):
                                    st.image(fig.get("path"), caption=fig.get("caption"))
                            with cols[1]:
                                st.json(fig)
                    
                    # Create ZIP file for download
                    import zipfile
                    zip_path = os.path.join(output_dir, "extracted_figures.zip")
                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        # Add metadata JSON
                        metadata_path = os.path.join(output_dir, "figures_metadata.json")
                        zipf.write(metadata_path, arcname="figures_metadata.json")
                        
                        # Add figure files
                        for fig in figures:
                            if fig.get("path") and os.path.exists(fig.get("path")):
                                zipf.write(fig.get("path"), arcname=fig.get("filename"))
                    
                    # Download button
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download All Figures (ZIP)",
                            data=f.read(),
                            file_name="extracted_figures.zip",
                            mime="application/zip"
                        )
                else:
                    st.warning("⚠️ No figures were found in the uploaded PDF")

# Run Streamlit app if executed directly
if __name__ == "__main__":
    run_figure_extractor_app() 