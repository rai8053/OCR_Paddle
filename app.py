"""
PaddleOCR RAG Web Application
Production-ready UI for PDF question answering
"""
import os
import tempfile
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
import gradio as gr
import logging
import sys

from pdf_parser import PaddleMathParser
from retriever import HybridRetriever
from qa_agent import MathQAAgent

# ---------------------------------------------------------------------------
# Console / logging encoding (Windows-safe)
# ---------------------------------------------------------------------------
# Some Windows terminals use cp1252 and cannot print emoji / some Unicode
# characters, which would normally raise UnicodeEncodeError when logging.
# We:
#   1. Reconfigure stdout / stderr to use UTF‑8 with a non-crashing handler.
#   2. Ensure the log file is always written as UTF‑8.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout.reconfigure(errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stderr.reconfigure(errors="replace")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paddle_rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaddleRAGApp:
    """Main application class with caching and error handling"""
    
    def __init__(self):
        self.parser = None
        self.retriever = None
        self.agent = None
        self.chunks = []
        self.current_pdf_name = None
        
        # Create cache directory
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Metrics
        self.metrics = {
            'pdfs_processed': 0,
            'questions_answered': 0,
            'cache_hits': 0
        }
    
    def initialize(self):
        """Lazy initialization of components"""
        if self.parser is None:
            logger.info("="*60)
            logger.info("PADDLEOCR RAG SYSTEM - CPU MODE")
            logger.info("="*60)
            
            try:
                self.parser = PaddleMathParser(dpi=150)
                self.retriever = HybridRetriever()
                logger.info("System initialized successfully")
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                raise
    
    def get_pdf_hash(self, file_path: str) -> str:
        """Generate hash for caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def process_pdf(self, file_obj, progress=gr.Progress()):
        """Handle PDF upload, OCR, and indexing with caching"""
        self.initialize()
        
        if file_obj is None:
            return "❌ Please upload a PDF file", None, self._get_stats()
        
        try:
            filename = Path(file_obj.name).name
            logger.info(f"Processing PDF: {filename}")
            
            # Calculate PDF hash for caching
            pdf_hash = self.get_pdf_hash(file_obj.name)
            cache_file = self.cache_dir / f"{pdf_hash}.json"
            
            # Check cache first
            if cache_file.exists():
                progress(0.5, desc=f"📦 Loading from cache...")
                logger.info(f"Cache hit for {filename}")
                
                self.chunks = self.parser.load_chunks(str(cache_file))
                self.metrics['cache_hits'] += 1
                
                if not self.chunks:
                    raise ValueError("Cache file is empty or corrupted")
                
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    shutil.copyfile(file_obj.name, tmp.name)
                    pdf_path = tmp.name
                
                try:
                    # Step 1: OCR with PaddleOCR
                    progress(0.2, desc=f"🔍 OCR Processing {filename}...")
                    self.chunks = self.parser.parse_pdf(pdf_path)
                    
                    if not self.chunks:
                        raise ValueError("No text extracted from PDF")
                    
                    # Save to cache
                    self.parser.save_chunks(self.chunks, str(cache_file))
                    logger.info(f"Saved to cache: {cache_file}")
                    
                finally:
                    # Cleanup temporary file
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
            
            # Step 2: Index chunks for search
            progress(0.6, desc=f"🔗 Indexing {len(self.chunks)} chunks...")
            self.retriever.clear()  # Clear previous documents
            self.retriever.add_documents(self.chunks)
            
            # Step 3: Initialize QA agent
            progress(0.9, desc="🤖 Initializing QA system...")
            openai_key = os.getenv('OPENAI_API_KEY')
            self.agent = MathQAAgent(self.retriever, openai_key)
            
            # Update metrics
            self.metrics['pdfs_processed'] += 1
            self.current_pdf_name = filename
            
            # Generate success message
            num_pages = len(set(c.metadata['page'] for c in self.chunks))
            avg_confidence = sum(
                c.metadata.get('confidence', 0) for c in self.chunks
            ) / len(self.chunks)
            
            success_msg = f"""✅ **Ready!** 
            
📄 **File:** {filename}
📊 **Stats:**
  - {len(self.chunks)} chunks indexed
  - {num_pages} pages processed
  - {avg_confidence:.2%} avg OCR confidence
  
💬 You can now ask questions!"""
            
            logger.info(f"Successfully processed {filename}")
            return success_msg, None, self._get_stats()
            
        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            logger.error(f"Processing failed: {e}", exc_info=True)
            return error_msg, None, self._get_stats()
    
    def ask_question(self, question, history):
        """Handle user questions"""
        if not question or not question.strip():
            return "", history, self._get_stats()
        
        if self.agent is None:
            history.append((question, "⚠️ Please upload a PDF first"))
            return "", history, self._get_stats()
        
        try:
            logger.info(f"Question: {question}")
            
            # Get answer from agent
            result = self.agent.ask(question, k=5)
            
            # Format answer with citations
            answer = result['answer']
            
            if result['citations']:
                answer += "\n\n**📚 Sources:**"
                for i, cit in enumerate(result['citations'], 1):
                    page = cit['page']
                    snippet = cit['text'][:100] + "..." if len(cit['text']) > 100 else cit['text']
                    confidence = cit.get('confidence', 0)
                    
                    answer += f"\n{i}. **Page {page}**"
                    if confidence > 0:
                        answer += f" (conf: {confidence:.2f})"
                    answer += f"\n   _{snippet}_\n"
            
            history.append((question, answer))
            self.metrics['questions_answered'] += 1
            
            logger.info(f"Answer generated with {len(result['citations'])} citations")
            return "", history, self._get_stats()
            
        except Exception as e:
            error_answer = f"❌ Error: {str(e)}"
            logger.error(f"Question answering failed: {e}", exc_info=True)
            history.append((question, error_answer))
            return "", history, self._get_stats()
    
    def export_conversation(self, history):
        """Export conversation to markdown file"""
        if not history:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.md"
            filepath = self.cache_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# PDF Q&A Conversation\n\n")
                f.write(f"**PDF:** {self.current_pdf_name or 'Unknown'}\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Exchanges:** {len(history)}\n\n")
                f.write("---\n\n")
                
                for i, (q, a) in enumerate(history, 1):
                    f.write(f"## Question {i}\n\n")
                    f.write(f"**Q:** {q}\n\n")
                    f.write(f"**A:** {a}\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Conversation exported to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def _get_stats(self):
        """Get current system statistics"""
        stats = f"""**System Stats:**
- PDFs processed: {self.metrics['pdfs_processed']}
- Questions answered: {self.metrics['questions_answered']}
- Cache hits: {self.metrics['cache_hits']}"""
        
        if self.retriever:
            retriever_stats = self.retriever.get_stats()
            stats += f"\n- Indexed docs: {retriever_stats['total_documents']}"
        
        return stats

# Create global app instance
app = PaddleRAGApp()

# Build Gradio interface
with gr.Blocks(
    title="PaddleOCR Math Textbook RAG", 
    theme=gr.themes.Soft(),
    css="""
        footer {visibility: hidden}
        .stats-box {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }
    """
) as demo:
    
    gr.Markdown("""
    # 📚 Mathematics Textbook Q&A with PaddleOCR
    ### Pure CPU Implementation - No GPU Required
    
    Upload your PDF textbook and ask questions about the content.
    
    **Features:**
    - 🔍 **PaddleOCR** for high-accuracy text recognition
    - 📄 **Page-level citations** for every answer  
    - ⚡ **Hybrid search** (semantic + keyword matching)
    - 💾 **Smart caching** for faster reprocessing
    - 💻 **Runs entirely on CPU**
    """)
    
    with gr.Row():
        # Left Column - Upload & Status
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📎 Upload PDF Textbook",
                file_types=[".pdf"],
                file_count="single"
            )
            
            status = gr.Textbox(
                label="Status",
                value="Ready. Upload a PDF to begin.",
                interactive=False,
                lines=8
            )
            
            stats = gr.Markdown(
                "**System Stats:** No data yet",
                elem_classes=["stats-box"]
            )
            
            gr.Markdown("""
            ### ⏱️ Processing Time
            - **First run**: Model download (2-3 min)
            - **Subsequent runs**: Instant (cached)
            - **PDF parsing**: 3-5 sec/page
            - **Indexing**: 1-2 sec/100 chunks
            - **Queries**: <1 second
            
            ### 📊 Supported Content
            - Printed text & handwriting
            - Mathematical equations
            - Tables and diagrams
            - Mixed layouts
            
            ### 💡 Tips
            - Higher DPI = better quality
            - Cache speeds up reprocessing
            - Use specific questions
            """)
        
        # Right Column - Chat Interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Ask Questions",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the definition of sample variance on page 53?",
                lines=2
            )
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", size="sm")
                export_btn = gr.Button("📥 Export", size="sm")
                submit_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
            
            # Example questions
            gr.Examples(
                examples=[
                    "What is the central limit theorem?",
                    "Explain the formula for variance",
                    "Summarize the content on page 10",
                    "What are the key concepts in chapter 3?"
                ],
                inputs=question,
                label="💡 Example Questions"
            )
            
            export_file = gr.File(
                label="Exported Conversation",
                visible=False
            )
    
    # Event handlers
    pdf_input.upload(
        fn=app.process_pdf,
        inputs=pdf_input,
        outputs=[status, chatbot, stats]
    )
    
    submit_btn.click(
        fn=app.ask_question,
        inputs=[question, chatbot],
        outputs=[question, chatbot, stats]
    )
    
    question.submit(
        fn=app.ask_question,
        inputs=[question, chatbot],
        outputs=[question, chatbot, stats]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, question]
    )
    
    export_btn.click(
        fn=app.export_conversation,
        inputs=[chatbot],
        outputs=[export_file]
    )

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 STARTING PADDLEOCR RAG SYSTEM")
    logger.info("💻 CPU MODE - Python 3.10/3.11")
    logger.info("="*60)
    
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        show_error=True,
        show_api=False
    )