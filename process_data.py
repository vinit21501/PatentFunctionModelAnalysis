import os
import fitz  # PyMuPDF
from PIL import Image
import logging
import io
import json
import base64
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def extract_pdf_content(pdf_path):
    """Extract text, metadata, and images from PDF using PyMuPDF."""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        structured_text = []
        images_data = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Extract raw text from page
                page_text = page.get_text()
                structured_text.append({
                    'page_number': page_num + 1,
                    'text': page_text
                })
                
                # Extract images with metadata
                try:
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Convert image to base64 string
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            
                            # Generate image embedding
                            # img_embedding = get_image_embedding(image)
                            
                            # Store image data with metadata
                            images_data.append({
                                'page_number': page_num + 1,
                                'image_index': img_index,
                                'base64': img_str,
                                # 'embedding': img_embedding,
                                'metadata': {
                                    'width': image.width,
                                    'height': image.height,
                                    'format': image.format,
                                    'mode': image.mode
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Error processing image {img_index} on page {page_num + 1}: {str(e)}")
                            continue
                except Exception as e:
                    logger.warning(f"Error extracting images from page {page_num + 1}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                continue
        
        return {
            'structured_text': structured_text,
            'images': images_data,
            'metadata': {
                'page_count': len(doc),
                'image_count': len(images_data),
                'file_name': os.path.basename(pdf_path)
            }
        }
    except Exception as e:
        logger.error(f"Error extracting PDF content: {str(e)}")
        return None
    finally:
        if doc:
            doc.close()

def main():
    pass

if __name__ == "__main__":
    main()