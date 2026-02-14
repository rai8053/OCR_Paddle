"""
Enhanced PaddleOCR PDF Parser
CPU-Optimized Version - Python 3.10/3.11 Compatible
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import time
from pathlib import Path

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
    """PDF parser using PaddleOCR - CPU optimized with comprehensive error handling"""
    
    def __init__(self, dpi: int = 150, cpu_threads: int = 4):
        """
        Initialize PaddleOCR parser with CPU optimizations
        
        Args:
            dpi: Resolution for PDF to image conversion (lower = faster, less memory)
            cpu_threads: Number of CPU threads to use
        """
        logger.info("=" * 50)
        logger.info("Initializing PaddleOCR (CPU Mode)")
        logger.info("=" * 50)
        
        self.dpi = dpi
        self.cpu_threads = cpu_threads
        
        # Performance tracking
        self.stats = {
            'pages_processed': 0,
            'chunks_generated': 0,
            'avg_confidence': 0.0,
            'processing_time': 0.0
        }
        
        try:
            # MINIMAL PaddleOCR configuration - only essential parameters
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang='en',            # English language
                use_gpu=False         # Force CPU mode
                # ALL OTHER PARAMETERS REMOVED for maximum compatibility
            )
            
            logger.info(f"✓ PaddleOCR initialized successfully")
            logger.info(f"  - Mode: CPU-only")
            logger.info(f"  - DPI: {dpi}")
            
        except ImportError as e:
            logger.error(f"Failed to import PaddleOCR: {e}")
            logger.error("Please install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str, Optional[int]]:
        """
        Validate PDF before processing
        
        Returns:
            Tuple of (is_valid, message, page_count)
        """
        MAX_SIZE = 100 * 1024 * 1024  # 100MB
        MAX_PAGES = 500
        
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                return False, f"File not found: {pdf_path}", None
            
            # Check file size
            file_size = os.path.getsize(pdf_path)
            if file_size > MAX_SIZE:
                return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)", None
            
            if file_size == 0:
                return False, "File is empty", None
            
            # Check PDF validity and page count
            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                
                if page_count == 0:
                    doc.close()
                    return False, "PDF has no pages", page_count
                
                if page_count > MAX_PAGES:
                    doc.close()
                    return False, f"Too many pages: {page_count} (max {MAX_PAGES})", page_count
                
                if doc.is_encrypted:
                    doc.close()
                    return False, "Encrypted PDF not supported", page_count
                
                # Check if PDF is readable
                try:
                    # Try to load first page
                    doc.load_page(0)
                except Exception:
                    doc.close()
                    return False, "PDF appears corrupted or unreadable", page_count
                
                doc.close()
                return True, f"Valid PDF with {page_count} pages", page_count
                
            except fitz.FileDataError:
                return False, "Invalid or corrupted PDF file", None
            except Exception as e:
                return False, f"PDF validation error: {str(e)}", None
            
        except Exception as e:
            return False, f"Validation failed: {str(e)}", None
    
    def convert_pdf_to_images(self, pdf_path: str, max_pages: Optional[int] = None) -> List[Image.Image]:
        """
        Convert PDF to images using PyMuPDF with memory optimization
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to convert (None for all)
        """
        logger.info(f"Converting PDF to images (DPI: {self.dpi})...")
        images = []
        start_time = time.time()
        
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            pages_to_convert = min(total_pages, max_pages) if max_pages else total_pages
            
            logger.info(f"  Total pages: {total_pages}, Converting: {pages_to_convert}")
            
            for page_num in range(pages_to_convert):
                try:
                    page = pdf_document[page_num]
                    
                    # Convert page to image with memory-efficient settings
                    zoom = self.dpi / 72
                    matrix = fitz.Matrix(zoom, zoom)
                    
                    # Use alpha=False to save memory
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                    
                    # Log progress every 10 pages
                    if (page_num + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"  Converted {page_num + 1}/{pages_to_convert} pages ({elapsed:.1f}s)")
                    
                    # Clear pixmap to free memory
                    del pix
                    
                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Converted {len(images)} pages in {elapsed:.1f}s")
            
            return images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            # Clean up any partial images
            for img in images:
                img.close()
            raise
    
    def parse_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> List[DocumentChunk]:
        """
        Extract text from PDF using PaddleOCR with CPU optimizations
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process (None for all)
        """
        start_time = time.time()
        filename = os.path.basename(pdf_path)
        
        logger.info(f"📄 Processing PDF: {filename}")
        
        # Validate PDF first
        valid, message, total_pages = self.validate_pdf(pdf_path)
        if not valid:
            raise ValueError(f"Invalid PDF: {message}")
        
        if total_pages:
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            logger.info(f"  Pages to process: {pages_to_process}/{total_pages}")
        
        # Convert PDF to images
        try:
            images = self.convert_pdf_to_images(pdf_path, max_pages)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
        
        all_chunks = []
        total_confidence = 0.0
        pages_with_text = 0
        
        try:
            for page_num, image in enumerate(images):
                logger.info(f"🔍 OCR Page {page_num + 1}/{len(images)}...")
                page_start = time.time()
                
                try:
                    # Convert PIL image to numpy array
                    img_np = np.array(image)
                    
                    # Run OCR on the page with retry logic
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            result = self.ocr.ocr(img_np, cls=True)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            logger.warning(f"  OCR attempt {attempt + 1} failed, retrying...")
                            time.sleep(1)
                    
                    # Extract text and confidence scores
                    texts = []
                    confidences = []
                    
                    if result and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line and len(line) > 1:
                                text = line[1][0].strip()
                                confidence = float(line[1][1])
                                
                                # Filter low confidence and very short text
                                if text and len(text) >= 3 and confidence >= 0.3:
                                    texts.append(text)
                                    confidences.append(confidence)
                    
                    # Process extracted text
                    if texts:
                        full_text = "\n".join(texts)
                        avg_confidence = sum(confidences) / len(confidences)
                        total_confidence += avg_confidence
                        pages_with_text += 1
                        
                        # Split page into chunks
                        chunks = self._smart_split_text(
                            text=full_text,
                            page_num=page_num,
                            confidence=avg_confidence,
                            chunk_size=500,
                            overlap=100
                        )
                        
                        all_chunks.extend(chunks)
                        
                        page_time = time.time() - page_start
                        logger.info(f"  ✓ Page {page_num + 1}: {len(chunks)} chunks, "
                                  f"conf: {avg_confidence:.2f}, time: {page_time:.1f}s")
                    else:
                        logger.info(f"  ⚠ Page {page_num + 1}: No text found")
                    
                except Exception as e:
                    logger.error(f"  ❌ OCR failed on page {page_num + 1}: {e}")
                    continue
                
                finally:
                    # Clear numpy array to free memory
                    del img_np
                    
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            raise
            
        finally:
            # Cleanup images
            for img in images:
                img.close()
        
        # Update statistics
        elapsed = time.time() - start_time
        self.stats['pages_processed'] = len(images)
        self.stats['chunks_generated'] = len(all_chunks)
        self.stats['avg_confidence'] = total_confidence / pages_with_text if pages_with_text else 0
        self.stats['processing_time'] = elapsed
        
        logger.info("=" * 50)
        logger.info(f"✅ Processing Complete: {filename}")
        logger.info(f"  Pages processed: {len(images)}")
        logger.info(f"  Chunks generated: {len(all_chunks)}")
        logger.info(f"  Avg confidence: {self.stats['avg_confidence']:.2f}")
        logger.info(f"  Total time: {elapsed:.1f}s")
        logger.info("=" * 50)
        
        return all_chunks
    
    def _smart_split_text(self, text: str, page_num: int, 
                         confidence: float = 0.0,
                         chunk_size: int = 500, 
                         overlap: int = 100,
                         min_chunk_size: int = 50) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks with smart boundary detection
        
        Args:
            text: Text to split
            page_num: Page number
            confidence: OCR confidence score
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        if not text:
            return chunks
        
        # If text is short enough, keep as one chunk
        if len(text) <= chunk_size:
            chunks.append(DocumentChunk(
                text=text,
                metadata={
                    'page': page_num + 1,  # 1-based page numbers
                    'source': 'pdf',
                    'engine': 'paddleocr',
                    'confidence': round(confidence, 3),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'char_count': len(text)
                },
                chunk_id=f"p{page_num + 1:03d}_c000"
            ))
            return chunks
        
        # Split with overlap to preserve context
        start = 0
        chunk_idx = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for natural break points
                for delimiter in ['. ', '.\n', '! ', '? ', '\n\n', '.', '\n']:
                    last_delim = text.rfind(delimiter, start, end)
                    if last_delim > start + chunk_size // 2:  # At least halfway through
                        end = last_delim + len(delimiter)
                        break
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Only keep chunks that meet minimum size
            if chunk_text and len(chunk_text) >= min_chunk_size:
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        'page': page_num + 1,  # 1-based page numbers
                        'source': 'pdf',
                        'engine': 'paddleocr',
                        'confidence': round(confidence, 3),
                        'chunk_index': chunk_idx,
                        'char_count': len(chunk_text),
                        'start_pos': start,
                        'end_pos': end
                    },
                    chunk_id=f"p{page_num + 1:03d}_c{chunk_idx:03d}"
                ))
                chunk_idx += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """
        Save chunks to JSON file for caching
        
        Args:
            chunks: List of DocumentChunk objects
            output_path: Path to save JSON file
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Convert chunks to serializable format
            data = []
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                # Ensure metadata is JSON serializable
                chunk_dict['metadata'] = {
                    k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v))
                    for k, v in chunk_dict['metadata'].items()
                }
                data.append(chunk_dict)
            
            # Save with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_path) / 1024  # KB
            logger.info(f"✓ Saved {len(chunks)} chunks to {output_path} ({file_size:.1f} KB)")
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            raise
    
    def load_chunks(self, input_path: str) -> List[DocumentChunk]:
        """
        Load chunks from JSON file
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            if not os.path.exists(input_path):
                logger.warning(f"Cache file not found: {input_path}")
                return []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error(f"Invalid cache format: expected list, got {type(data)}")
                return []
            
            chunks = []
            for item in data:
                try:
                    chunk = DocumentChunk(
                        text=item['text'],
                        metadata=item['metadata'],
                        chunk_id=item['chunk_id']
                    )
                    chunks.append(chunk)
                except KeyError as e:
                    logger.warning(f"Skipping invalid chunk (missing key: {e})")
                    continue
            
            file_size = os.path.getsize(input_path) / 1024  # KB
            logger.info(f"✓ Loaded {len(chunks)} chunks from {input_path} ({file_size:.1f} KB)")
            return chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics"""
        return self.stats.copy()
    
    def clear_stats(self):
        """Clear parser statistics"""
        self.stats = {
            'pages_processed': 0,
            'chunks_generated': 0,
            'avg_confidence': 0.0,
            'processing_time': 0.0
        }