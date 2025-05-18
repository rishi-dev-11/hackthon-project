import os
import cv2
import numpy as np
from PIL import Image
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize PaddleOCR
def initialize_paddle_ocr():
    """Initialize PaddleOCR with handwritten text recognition capability."""
    try:
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR with support for multiple languages and handwritten text
        # use_angle_cls for detecting text orientation
        # use_gpu to utilize GPU if available
        # lang can be set to 'en' for English, 'ch' for Chinese, etc.
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        logger.info("PaddleOCR initialized successfully")
        return paddle_ocr
    except ImportError:
        logger.error("PaddleOCR is not installed. Please install with: pip install paddlepaddle paddleocr")
        return None
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {e}", exc_info=True)
        return None

# Process image with PaddleOCR
def process_image_with_paddle_ocr(image_path=None, image_bytes=None):
    """Extract text from an image using PaddleOCR, optimized for handwritten text."""
    try:
        # Initialize PaddleOCR
        paddle_ocr = initialize_paddle_ocr()
        if paddle_ocr is None:
            return "Error: PaddleOCR initialization failed"

        # Load image either from path or bytes
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
        elif image_bytes:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return "Error: No valid image path or bytes provided"

        # Check if image is loaded properly
        if img is None:
            return "Error: Failed to load image"

        # Preprocess image for better OCR results
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply adaptive thresholding to improve text detection
        # Different methods for different image qualities
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. Noise removal with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 4. Save preprocessed image temporarily
        temp_path = "temp_processed_image.jpg"
        cv2.imwrite(temp_path, opening)
        
        # Run OCR on both original and preprocessed images
        result_original = paddle_ocr.ocr(img, cls=True)
        result_processed = paddle_ocr.ocr(temp_path, cls=True)
        
        # Combine results and extract text
        extracted_text = []
        confidence_threshold = 0.5  # Minimum confidence level to accept
        
        # Process results from original image
        if result_original and len(result_original) > 0:
            for line in result_original[0]:  # Assuming the first element contains detection results
                if line and len(line) >= 2:
                    text, confidence = line[1]
                    if confidence >= confidence_threshold:
                        extracted_text.append(text)
        
        # Process results from preprocessed image
        if result_processed and len(result_processed) > 0:
            for line in result_processed[0]:  # Assuming the first element contains detection results
                if line and len(line) >= 2:
                    text, confidence = line[1]
                    if confidence >= confidence_threshold and text not in extracted_text:
                        extracted_text.append(text)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if not extracted_text:
            return "No text detected in the image"
        
        # Join all detected text
        full_text = " ".join(extracted_text)
        logger.info(f"PaddleOCR extracted text: {full_text[:100]}...")
        return full_text
        
    except Exception as e:
        logger.error(f"Error processing image with PaddleOCR: {e}", exc_info=True)
        return f"Error processing image with PaddleOCR: {str(e)}"

# Process figure directly from the application
def process_figure_with_paddle_ocr(figure_data):
    """Process a figure object from the application using PaddleOCR."""
    try:
        if not figure_data:
            return "No figure data provided"
            
        # Extract image bytes from figure data
        if "image_bytes" not in figure_data or not figure_data["image_bytes"]:
            return "No image data in figure"
            
        image_bytes = figure_data["image_bytes"]
        extracted_text = process_image_with_paddle_ocr(image_bytes=image_bytes)
        
        return extracted_text
    except Exception as e:
        logger.error(f"Error processing figure: {e}", exc_info=True)
        return f"Error processing figure: {str(e)}" 