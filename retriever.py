"""
Hybrid Retrieval System - CPU optimized
Combines FAISS vector search + BM25 keyword search
"""
import logging
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid search using both vector similarity and keyword matching"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info("Initializing Hybrid Retriever (CPU mode)...")
        
        try:
            # Use BAAI/bge-small-en for embeddings (CPU optimized)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            self.vectorstore = None
            self.documents = []
            self.bm25_retriever = None
            
            logger.info("Hybrid Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise
    
    def add_documents(self, chunks: List[Any]):
        """Index documents for both dense and sparse retrieval"""
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        try:
            # Convert to LangChain Document format
            docs = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.text[:2000],  # Limit length for efficiency
                    metadata=chunk.metadata
                )
                docs.append(doc)
            
            self.documents.extend(docs)
            
            # 1. Dense retriever (FAISS vector similarity)
            logger.info("Building FAISS index...")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vectorstore.add_documents(docs)
            
            # 2. Sparse retriever (BM25 keyword search)
            logger.info("Building BM25 index...")
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 10
            
            logger.info(f"Indexing complete. Total documents: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Hybrid search: combine vector + keyword results"""
        if not self.vectorstore or not self.bm25_retriever:
            logger.warning("Retriever not initialized. No documents indexed.")
            return []
        
        try:
            # Dense retriever (semantic similarity)
            dense = self.vectorstore.as_retriever(
                search_kwargs={"k": k * 2}  # Get more candidates
            )
            
            # Sparse retriever (keyword matching)
            self.bm25_retriever.k = k * 2
            
            # Ensemble: 70% semantic, 30% keyword
            ensemble = EnsembleRetriever(
                retrievers=[dense, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            # Retrieve and rank
            results = ensemble.invoke(query)
            
            # Deduplicate by chunk_id if available
            seen_ids = set()
            unique_results = []
            
            for doc in results:
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_results.append(doc)
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def clear(self):
        """Clear all indexed documents"""
        logger.info("Clearing retriever...")
        self.vectorstore = None
        self.documents = []
        self.bm25_retriever = None
    
    def get_stats(self) -> dict:
        """Get retriever statistics"""
        return {
            'total_documents': len(self.documents),
            'has_vectorstore': self.vectorstore is not None,
            'has_bm25': self.bm25_retriever is not None
        }