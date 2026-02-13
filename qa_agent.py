"""
QA Agent with Citation Support
Retrieves and formats answers from the textbook
"""
import logging
from typing import Dict, List
from time import time, sleep

logger = logging.getLogger(__name__)

class MathQAAgent:
    """Answer questions using retrieved context from PaddleOCR"""
    
    def __init__(self, retriever, openai_api_key: str = None):
        self.retriever = retriever
        self.use_openai = False
        self.last_request_time = 0
        self.min_interval = 1.0  # Rate limiting: 1 second between requests
        
        # Optional: Use OpenAI for better answers
        if openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                from langchain.prompts import ChatPromptTemplate
                
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    api_key=openai_api_key,
                    max_tokens=1000
                )
                
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful mathematics professor assistant.

Answer the question using ONLY the provided context from the textbook.
- Cite page numbers when referencing information
- Use LaTeX notation for math: inline $x^2$ or display $$\\int_0^1 x dx$$
- If the context doesn't contain the answer, say "I cannot find this information in the provided pages"
- Be concise but thorough
- Maintain academic tone"""),
                    ("human", """Context from textbook:
{context}

Question: {question}

Answer:""")
                ])
                
                self.use_openai = True
                logger.info("OpenAI LLM enabled (gpt-4o-mini)")
            except ImportError:
                logger.warning("OpenAI not available (langchain-openai not installed)")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        logger.info(f"QA Agent ready (LLM: {'OpenAI' if self.use_openai else 'Retrieval-only'})")
    
    def ask(self, question: str, k: int = 5) -> Dict:
        """Answer question by retrieving and formatting context"""
        
        # Rate limiting for OpenAI
        if self.use_openai:
            elapsed = time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep(self.min_interval - elapsed)
        
        try:
            # 1. Retrieve relevant chunks
            docs = self.retriever.search(question, k=k)
            
            if not docs:
                return {
                    'question': question,
                    'answer': "No relevant information found. Please upload a PDF first.",
                    'citations': [],
                    'context': ''
                }
            
            # 2. Format context with citations
            context_parts = []
            citations = []
            seen_pages = set()
            
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get('page', '?')
                confidence = doc.metadata.get('confidence', 0.0)
                text = doc.page_content
                
                # Truncate long content for display
                display_text = text[:200] + "..." if len(text) > 200 else text
                
                context_parts.append(f"[Source {i} - Page {page}]:\n{text}")
                
                # Add to citations (avoid duplicate pages)
                if page not in seen_pages:
                    citations.append({
                        'page': page,
                        'text': display_text,
                        'confidence': confidence,
                        'source_num': i
                    })
                    seen_pages.add(page)
            
            context = "\n\n".join(context_parts)
            
            # 3. Generate answer
            if self.use_openai:
                try:
                    chain = self.prompt | self.llm
                    response = chain.invoke({
                        "context": context,
                        "question": question
                    })
                    answer = response.content
                    self.last_request_time = time()
                    
                except Exception as e:
                    logger.error(f"OpenAI API call failed: {e}")
                    answer = self._format_retrieval_answer(context, citations)
            else:
                answer = self._format_retrieval_answer(context, citations)
            
            return {
                'question': question,
                'answer': answer,
                'citations': citations[:5],  # Limit to top 5
                'context': context[:1000]  # Truncated context for debugging
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'citations': [],
                'context': ''
            }
    
    def _format_retrieval_answer(self, context: str, citations: List[Dict]) -> str:
        """Format answer when not using LLM"""
        answer = "**Retrieved from textbook:**\n\n"
        
        # Show top 3 most relevant excerpts
        for i, cit in enumerate(citations[:3], 1):
            page = cit['page']
            text = cit['text']
            confidence = cit.get('confidence', 0.0)
            
            answer += f"**{i}. Page {page}** "
            if confidence > 0:
                answer += f"(confidence: {confidence:.2f})"
            answer += f"\n{text}\n\n"
        
        if len(citations) > 3:
            other_pages = [str(c['page']) for c in citations[3:]]
            answer += f"\n*Also see pages: {', '.join(other_pages)}*"
        
        return answer
    
    def get_conversation_summary(self, history: List[tuple]) -> str:
        """Generate summary of conversation history"""
        if not history:
            return "No conversation history"
        
        summary = f"**Conversation Summary** ({len(history)} exchanges)\n\n"
        for i, (q, a) in enumerate(history[-5:], 1):  # Last 5 exchanges
            summary += f"{i}. Q: {q[:100]}...\n"
        
        return summary