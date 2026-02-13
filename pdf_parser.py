"""
Enhanced PaddleOCR PDF Parser
Python 3.10/3.11 Compatible - Production Ready
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a single chunk of document with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

class PaddleMathParser:
    """PDF parser using PaddleOCR - CPU optimized with error handling"""
    
    def __init__(self, dpi: int = 150):
        logger.info("Initializing PaddleOCR 3.x (CPU mode)...")
        self.dpi = dpi
        
        try:
            # Initialize PaddleOCR with CPU optimizations
            self.ocr = PaddleOCR(
                use_angle_cls=True,      # Enable angle classification
                lang='en',               # English language
                show_log=False,          # Reduce console noise
                use_gpu=False,           # Force CPU mode
                cpu_threads=4,           # Use 4 CPU threads
                det_db_thresh=0.3,       # Detection threshold
                det_db_box_thresh=0.2,   # Box threshold
                use_tensorrt=False,      # No GPU optimizations
                enable_mkldnn=True       # Intel CPU optimization
            )
            logger.info("PaddleOCR 3.x ready")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Validate PDF before processing"""
        MAX_SIZE = 100 * 1024 * 1024  # 100MB
        MAX_PAGES = 500
        
        try:
            # Check file exists and size
            if not os.path.exists(pdf_path):
                return False, "File does not exist"
            
            file_size = os.path.getsize(pdf_path)
            if file_size > MAX_SIZE:
                return False, f"File too large ({file_size / 1024 / 1024:.1f}MB, max 100MB)"
            
            # Check PDF validity
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            if page_count == 0:
                doc.close()
                return False, "PDF has no pages"
            
            if page_count > MAX_PAGES:
                doc.close()
                return False, f"Too many pages ({page_count}, max {MAX_PAGES})"
            
            if doc.is_encrypted:
                doc.close()
                return False, "Encrypted PDFs not supported"
            
            doc.close()
            return True, "Valid PDF"
            
        except Exception as e:
            return False, f"Invalid PDF: {str(e)}"
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF"""
        logger.info("Converting PDF to images...")
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                    
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Converted {page_num + 1}/{len(pdf_document)} pages")
                        
                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            logger.info(f"Converted {len(images)} pages successfully")
            return images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def parse_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Main method: Extract text from PDF using PaddleOCR"""
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        
        # Validate PDF first
        valid, message = self.validate_pdf(pdf_path)
        if not valid:
            raise ValueError(f"Invalid PDF: {message}")
        
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)
        all_chunks = []
        
        try:
            for page_num, image in enumerate(images):
                logger.info(f"OCR Page {page_num + 1}/{len(images)}...")
                
                try:
                    # Convert PIL image to numpy array
                    img_np = np.array(image)
                    
                    # Run OCR on the page
                    result = self.ocr.ocr(img_np, cls=True)
                    
                    # Extract text from OCR result
                    texts = []
                    confidences = []
                    
                    if result and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line and len(line) > 1:
                                text = line[1][0].strip()
                                confidence = float(line[1][1])
                                
                                # Filter low confidence and very short text
                                if text and len(text) > 3 and confidence > 0.5:
                                    texts.append(text)
                                    confidences.append(confidence)
                    
                    # Join all text from the page
                    full_text = "\n".join(texts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    if full_text.strip():
                        # Split long pages into overlapping chunks
                        chunks = self._smart_split_text(
                            full_text, 
                            page_num, 
                            avg_confidence
                        )
                        all_chunks.extend(chunks)
                        logger.info(f"  -> Generated {len(chunks)} chunks (avg conf: {avg_confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"OCR failed on page {page_num + 1}: {e}")
                    continue
                    
        finally:
            # Cleanup images
            for img in images:
                img.close()
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(images)} pages")
        return all_chunks
    
    def _smart_split_text(self, text: str, page_num: int, 
                         confidence: float = 0.0,
                         chunk_size: int = 500, 
                         overlap: int = 100) -> List[DocumentChunk]:
        """Split text into overlapping chunks with smart boundary detection"""
        chunks = []
        
        # If text is short enough, keep as one chunk
        if len(text) <= chunk_size:
            chunks.append(DocumentChunk(
                text=text,
                metadata={
                    'page': page_num,
                    'source': 'pdf',
                    'engine': 'paddleocr',
                    'confidence': round(confidence, 3)
                },
                chunk_id=f"p{page_num}_c0"
            ))
            return chunks
        
        # Split with overlap to preserve context
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for delimiter in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_delim = text.rfind(delimiter, start, end)
                    if last_delim > start + chunk_size // 2:  # At least halfway through
                        end = last_delim + len(delimiter)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) > 20:  # Minimum chunk size
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        'page': page_num,
                        'source': 'pdf',
                        'engine': 'paddleocr',
                        'confidence': round(confidence, 3),
                        'chunk_index': chunk_idx
                    },
                    chunk_id=f"p{page_num}_c{chunk_idx}"
                ))
                chunk_idx += 1
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """Save chunks to JSON file for caching"""
        try:
            data = [asdict(chunk) for chunk in chunks]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
    
    def load_chunks(self, input_path: str) -> List[DocumentChunk]:
        """Load chunks from JSON file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = [
                DocumentChunk(
                    text=item['text'],
                    metadata=item['metadata'],
                    chunk_id=item['chunk_id']
                )
                for item in data
            ]
            
            logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []