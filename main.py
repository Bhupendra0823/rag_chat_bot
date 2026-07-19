# main.py - FastAPI Backend with RAG capabilities only

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Iterator, Tuple, TypedDict, Set
import os
import re
import asyncio
import hashlib
import json
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
import concurrent.futures
import logging
import sys
from logging.handlers import RotatingFileHandler
import queue
import threading
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# ==================== LOGGING CONFIGURATION ====================
class LogQueueHandler(logging.Handler):
    """Custom logging handler that sends logs to a queue for WebSocket streaming"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.log_queue.put({
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'line': record.lineno
            })
        except Exception:
            self.handleError(record)

# Global log queue for WebSocket streaming
log_queue = queue.Queue(maxsize=1000)
active_websockets: Set[WebSocket] = set()

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (for all logs)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler (only errors and above)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Queue handler for WebSocket streaming
    queue_handler = LogQueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(simple_formatter)
    logger.addHandler(queue_handler)
    
    # Create specific loggers for different components
    loggers = {
        'rag': logging.getLogger('rag'),
        'api': logging.getLogger('api'),
        'scraper': logging.getLogger('scraper'),
        'cache': logging.getLogger('cache'),
        'websocket': logging.getLogger('websocket'),
    }
    
    for log in loggers.values():
        log.setLevel(logging.INFO)
    
    return logger, loggers

# Initialize logging
logger, loggers = setup_logging()
rag_logger = loggers['rag']
api_logger = loggers['api']
scraper_logger = loggers['scraper']
cache_logger = loggers['cache']
ws_logger = loggers['websocket']

# ==================== LANGCHAIN IMPORTS ====================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==================== LLM FUNCTIONS ====================
from llm_config import get_aws_embeddings, get_aws_llm

# ==================== WEB SCRAPING IMPORTS ====================
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page
import nest_asyncio

# ==================== FOR CACHING & SEARCH ====================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nest_asyncio.apply()

# ==================== FASTAPI APP INITIALIZATION ====================
app = FastAPI(title="RAG System API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bkumar0823.space",
        "https://bkumar-portfolio.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "all"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_logger.info("FastAPI application initialized with CORS middleware")

# ==================== RAG STATE ====================
class RAGState(TypedDict):
    """State schema for our RAG application"""
    question: str
    documents: List[Document]
    context: str
    answer: str
    needs_retrieval: bool
    from_cache: bool
    stream_callback: Optional[callable]
    expanded_queries: Optional[List[str]]

# ==================== RAG CACHE SYSTEM ====================
class SemanticCache:
    """Semantic Cache for RAG responses"""
    
    def __init__(self, cache_file: str = "data/response_cache.json", similarity_threshold: float = 0.85):
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self.cache = self._load_cache()
        self.embeddings = get_aws_embeddings()
        cache_logger.info(f"SemanticCache initialized with threshold {similarity_threshold}")
        
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                cache_logger.debug(f"Cache loaded from {self.cache_file}")
                return cache_data
            except Exception as e:
                cache_logger.error(f"Error loading cache: {e}")
                return {"queries": [], "responses": [], "embeddings": [], "timestamps": []}
        cache_logger.debug("No existing cache found, creating new")
        return {"queries": [], "responses": [], "embeddings": [], "timestamps": []}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            cache_logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            cache_logger.error(f"Error saving cache: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            cache_logger.error(f"Error generating embedding: {e}")
            import random
            return [random.random() for _ in range(1536)]
    
    def find_similar_query(self, query: str) -> Tuple[Optional[str], Optional[str], float]:
        if not self.cache["queries"]:
            return None, None, 0.0
        
        try:
            query_embedding = self._get_embedding(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)
            cached_embeddings = np.array(self.cache["embeddings"])
            
            similarities = cosine_similarity(query_embedding, cached_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            cache_logger.debug(f"Best similarity score: {best_score:.3f}")
            
            if best_score >= self.similarity_threshold:
                cache_logger.info(f"Cache HIT! Similarity: {best_score:.3f}")
                return (
                    self.cache["queries"][best_idx],
                    self.cache["responses"][best_idx],
                    best_score
                )
            
            cache_logger.debug(f"Cache MISS (best score: {best_score:.3f} < threshold)")
            return None, None, best_score
        except Exception as e:
            cache_logger.error(f"Error finding similar query: {e}")
            return None, None, 0.0
    
    def add_response(self, question: str, answer: str):
        try:
            embedding = self._get_embedding(question)
            
            self.cache["queries"].append(question)
            self.cache["responses"].append(answer)
            self.cache["embeddings"].append(embedding)
            self.cache["timestamps"].append(datetime.now().isoformat())
            
            if len(self.cache["queries"]) > 1000:
                cache_logger.info("Cache limit reached, removing oldest entries")
                self.cache["queries"] = self.cache["queries"][-1000:]
                self.cache["responses"] = self.cache["responses"][-1000:]
                self.cache["embeddings"] = self.cache["embeddings"][-1000:]
                self.cache["timestamps"] = self.cache["timestamps"][-1000:]
            
            self._save_cache()
            cache_logger.info(f"Response cached for question: {question[:50]}...")
        except Exception as e:
            cache_logger.error(f"Error adding response to cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        stats = {
            "total_entries": len(self.cache["queries"]),
            "oldest_entry": self.cache["timestamps"][0] if self.cache["timestamps"] else None,
            "newest_entry": self.cache["timestamps"][-1] if self.cache["timestamps"] else None
        }
        cache_logger.debug(f"Cache stats: {stats}")
        return stats
    
    def clear(self):
        self.cache = {"queries": [], "responses": [], "embeddings": [], "timestamps": []}
        self._save_cache()
        cache_logger.info("Cache cleared")

# ==================== RAG SPA SCRAPER ====================
class SPAScraper:
    """Improved scraper for React/SPA websites with proper metadata extraction"""
    
    def __init__(self, base_url: str, max_pages: int = 30, wait_time: int = 5000):
        self.base_url = base_url
        self.max_pages = max_pages
        self.wait_time = wait_time
        self.visited_urls = set()
        self.pages_to_scrape = []
        self.scraped_data = []
        scraper_logger.info(f"SPAScraper initialized with base_url: {base_url}, max_pages: {max_pages}")
        
    async def wait_for_content(self, page: Page, timeout: int = 30000):
        """Wait for React to render content with proper timeouts"""
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout)
            await page.wait_for_selector(
                "div, main, article, section, .content, .main, #root",
                state="attached",
                timeout=timeout
            )
            await page.wait_for_timeout(self.wait_time)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)
            scraper_logger.debug(f"Content loaded for page: {page.url}")
        except Exception as e:
            scraper_logger.warning(f"Content wait timeout: {e}")
    
    def _extract_section(self, text: str, section_keywords: List[str]) -> Optional[str]:
        """Extract specific section from text using keywords"""
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        for i, line in enumerate(lines):
            for keyword in section_keywords:
                if keyword.lower() in line.lower():
                    in_section = True
                    section_content.append(line)
                    break
            
            if in_section:
                is_next_section = False
                if i + 1 < len(lines):
                    for keyword in section_keywords:
                        if keyword.lower() in lines[i + 1].lower():
                            is_next_section = True
                            break
                
                if not is_next_section:
                    section_content.append(line)
                else:
                    break
        
        return '\n'.join(section_content) if section_content else None
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from page"""
        metadata = {
            'url': url,
            'title': soup.title.string if soup.title else url,
            'description': '',
            'keywords': '',
            'sections': {}
        }
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        return metadata
    
    def _extract_sections_from_content(self, content: str) -> Dict[str, str]:
        """Extract specific sections from content"""
        sections = {}
        
        section_patterns = {
            'education': ['education', 'b.tech', 'btech', 'bachelor', 'intermediate', 'high school', 'school', 'college', 'university', 'degree'],
            'experience': ['experience', 'work experience', 'professional', 'career', 'employment', 'job', 'role'],
            'projects': ['projects', 'project', 'portfolio', 'built', 'developed', 'created'],
            'skills': ['skills', 'technical skills', 'technologies', 'tech stack', 'languages', 'tools'],
            'achievements': ['achievements', 'awards', 'certifications', 'recognitions'],
            'summary': ['summary', 'about', 'profile', 'professional summary']
        }
        
        for section_name, keywords in section_patterns.items():
            section_content = self._extract_section(content, keywords)
            if section_content and len(section_content) > 20:
                sections[section_name] = section_content.strip()
        
        return sections
    
    async def extract_content(self, page: Page, url: str) -> Dict[str, Any]:
        """Extract meaningful content with metadata"""
        try:
            html = await page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            metadata = self._extract_metadata(soup, url)
            
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            content_selectors = [
                'main', 'article', '.content', '.main-content', 
                '.post-content', '.blog-content', '#main-content',
                '.container', '.page-content', '.resume-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.body
            
            text = main_content.get_text(separator='\n', strip=True) if main_content else ""
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            
            sections = self._extract_sections_from_content(text)
            
            metadata['sections'] = sections
            metadata['content'] = text
            metadata['word_count'] = len(text.split())
            metadata['section_count'] = len(sections)
            metadata['sections_found'] = list(sections.keys())
            
            scraper_logger.debug(f"Extracted content from {url}: {metadata['word_count']} words, {len(sections)} sections")
            return metadata
            
        except Exception as e:
            scraper_logger.error(f"Error extracting content from {url}: {e}")
            return {
                'url': url, 
                'title': url, 
                'content': '', 
                'word_count': 0,
                'sections': {},
                'sections_found': []
            }
    
    async def scrape_page(self, page: Page, url: str) -> Optional[Dict]:
        """Scrape a single page with improved handling"""
        try:
            scraper_logger.info(f"Scraping: {url}")
            
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await self.wait_for_content(page)
            
            content_data = await self.extract_content(page, url)
            
            if content_data['word_count'] < 50:
                scraper_logger.warning(f"Skipping {url} - too little content ({content_data['word_count']} words)")
                return None
            
            links = await page.evaluate('''
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links
                        .map(link => link.href)
                        .filter(href => href.startsWith(window.location.origin))
                        .filter(href => !href.includes('#') && !href.includes('?'));
                }
            ''')
            
            self.pages_to_scrape.extend(links)
            return content_data
            
        except Exception as e:
            scraper_logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def scrape_website(self) -> List[Dict]:
        """Main scraping method"""
        scraper_logger.info(f"Starting website scraping: {self.base_url}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            self.pages_to_scrape = [self.base_url]
            self.visited_urls = set()
            scraped_pages = []
            
            while self.pages_to_scrape and len(scraped_pages) < self.max_pages:
                current_url = self.pages_to_scrape.pop(0)
                
                if current_url.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    continue
                
                if current_url in self.visited_urls:
                    continue
                    
                if urlparse(current_url).netloc != urlparse(self.base_url).netloc:
                    continue
                
                self.visited_urls.add(current_url)
                content = await self.scrape_page(page, current_url)
                
                if content:
                    scraped_pages.append(content)
                
                if len(scraped_pages) >= self.max_pages:
                    scraper_logger.info(f"Reached max pages limit: {self.max_pages}")
                    break
            
            await browser.close()
            
            seen_hashes = set()
            unique_pages = []
            for page_data in scraped_pages:
                content_hash = hashlib.md5(page_data['content'].encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_pages.append(page_data)
            
            self.scraped_data = unique_pages
            scraper_logger.info(f"Scraping complete: {len(unique_pages)} unique pages out of {len(scraped_pages)} total")
            
            return unique_pages
    
    def scrape_sync(self) -> List[Document]:
        """Synchronous wrapper with proper event loop handling"""
        scraper_logger.info("Starting synchronous scraping")
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                scraped_data = loop.run_until_complete(self.scrape_website())
                loop.close()
            else:
                scraped_data = loop.run_until_complete(self.scrape_website())
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            scraped_data = loop.run_until_complete(self.scrape_website())
            loop.close()
        
        documents = []
        for data in scraped_data:
            sections = data.get('sections', {})
            section_text = ""
            if sections:
                section_text = "\n\nEXTRACTED SECTIONS:\n"
                for section_name, section_content in sections.items():
                    section_text += f"\n[{section_name.upper()}]\n{section_content}\n"
            
            full_text = f"Page: {data['title']}\n\n{section_text}\n\nFULL CONTENT:\n{data['content']}"
            
            doc = Document(
                page_content=full_text,
                metadata={
                    'source': data['url'],
                    'title': data['title'],
                    'sections_found': data.get('sections_found', []),
                    'word_count': data['word_count']
                }
            )
            documents.append(doc)
        
        scraper_logger.info(f"Created {len(documents)} documents from scraped data")
        return documents

# ==================== RAG DATA INGESTION ====================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Improved chunking with larger size for better context"""
    rag_logger.info(f"Chunking {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    rag_logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: List[Document]) -> Chroma:
    """Create vector store with proper persistence"""
    rag_logger.info("Creating vector store")
    embeddings = get_aws_embeddings()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./data/website_chroma_db"
    )
    
    rag_logger.info("Vector store created successfully")
    return vector_store

# ==================== RAG RETRIEVAL ====================
def expand_query(question: str) -> List[str]:
    """Expand query with different phrasings"""
    variations = [question]
    
    if any(word in question.lower() for word in ['education', 'study', 'college', 'school', 'btech', 'b.tech']):
        variations.extend([
            "education background",
            "academic qualification",
            "educational history",
            "degrees and certifications",
            "college and school education"
        ])
    
    if any(word in question.lower() for word in ['experience', 'work', 'job', 'career', 'role']):
        variations.extend([
            "work experience",
            "professional experience",
            "career history",
            "employment background",
            "job roles"
        ])
    
    if any(word in question.lower() for word in ['project', 'portfolio', 'built', 'developed']):
        variations.extend([
            "projects and portfolio",
            "personal projects",
            "professional projects",
            "work portfolio"
        ])
    
    rag_logger.debug(f"Expanded query: {len(variations)} variations")
    return variations

def retrieve_documents(state: RAGState) -> RAGState:
    """Improved retrieval with expanded queries"""
    rag_logger.info(f"Retrieving documents for question: {state['question'][:50]}...")
    
    embeddings = get_aws_embeddings()
    vector_store = Chroma(
        persist_directory="./data/website_chroma_db",
        embedding_function=embeddings
    )
    
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    expanded_queries = expand_query(state["question"])
    
    all_documents = []
    seen_content = set()
    
    for query in expanded_queries[:3]:
        docs = base_retriever.invoke(query)
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_documents.append(doc)
    
    docs = base_retriever.invoke(state["question"])
    for doc in docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            all_documents.append(doc)
    
    all_documents = all_documents[:8]
    rag_logger.info(f"Retrieved {len(all_documents)} unique documents")
    
    return {
        **state,
        "documents": all_documents,
        "needs_retrieval": False
    }

def format_context(state: RAGState) -> RAGState:
    """Format context with better structure"""
    if not state["documents"]:
        rag_logger.warning("No documents found for context")
        return {
            **state,
            "context": "No relevant content found from the website.",
            "needs_retrieval": False
        }
    
    context_parts = []
    for i, doc in enumerate(state["documents"], 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "Untitled")
        sections = doc.metadata.get("sections_found", [])
        
        section_info = f" (Sections: {', '.join(sections)})" if sections else ""
        
        context_parts.append(
            f"[SOURCE {i}: {title}{section_info}]\n"
            f"URL: {source}\n"
            f"CONTENT:\n{doc.page_content}\n"
        )
    
    context = "\n" + "=" * 80 + "\n".join(context_parts) + "\n" + "=" * 80
    rag_logger.debug(f"Formatted context with {len(state['documents'])} sources")
    
    return {
        **state,
        "context": context,
        "needs_retrieval": False
    }

# ==================== RAG GENERATION ====================
def generate_answer_streaming(state: RAGState, cache: Optional[SemanticCache] = None) -> Iterator[str]:
    """Generate answer with streaming support"""
    
    if cache:
        cached_question, cached_answer, similarity = cache.find_similar_query(state["question"])
        
        if cached_answer and similarity >= 0.85:
            rag_logger.info(f"Cache HIT! Similarity: {similarity:.2f}")
            yield f"[CACHED] "
            for chunk in cached_answer.split():
                yield chunk + " "
                import time
                time.sleep(0.02)
            state["answer"] = cached_answer
            state["from_cache"] = True
            return
    
    rag_logger.info(f"Generating new response for question: {state['question'][:50]}...")
    
    llm = get_aws_llm()
    
    prompt = f"""You are a helpful assistant that answers questions about Bhupendra's portfolio website. 
Use ONLY the context provided below to answer the question.

Context from website:
{state["context"]}

Question: {state["question"]}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer cannot be found in the context, say so
3. Be thorough - scan ALL the context carefully
4. For education questions, look for: B.Tech, Intermediate (ISC), High School (ISC)
5. For experience questions, look for: TCS, System Engineer, Leasing Monk
6. For project questions, look for the specific project names
7. Cite the source page when possible
8. Be concise but comprehensive

Answer:"""
    
    full_answer = ""
    chunk_count = 0
    for chunk in llm.stream(prompt):
        content = chunk.content
        full_answer += content
        chunk_count += 1
        yield content
    
    if cache:
        cache.add_response(state["question"], full_answer)
        rag_logger.info(f"Response cached ({len(full_answer)} chars, {chunk_count} chunks)")
    
    state["answer"] = full_answer
    state["from_cache"] = False
    rag_logger.info("Generation complete")

def generate_answer(state: RAGState, cache: Optional[SemanticCache] = None) -> RAGState:
    """Non-streaming version"""
    full_answer = ""
    for chunk in generate_answer_streaming(state, cache):
        full_answer += chunk
    
    return {
        **state,
        "answer": full_answer,
        "from_cache": "CACHED" in full_answer
    }

# ==================== RAG GRAPH BUILDER ====================
def build_rag_graph(cache: Optional[SemanticCache] = None):
    """Build the complete RAG workflow"""
    rag_logger.info("Building RAG graph")
    
    def generate_with_cache(state: RAGState) -> RAGState:
        return generate_answer(state, cache)
    
    workflow = StateGraph(RAGState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("format", format_context)
    workflow.add_node("generate", generate_with_cache)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format")
    workflow.add_edge("format", "generate")
    workflow.add_edge("generate", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    rag_logger.info("RAG graph built successfully")
    return app

# ==================== RAG APP CLASS ====================
class RAGApp:
    def __init__(self, website_url: str = "https://bkumar0823.space/", max_pages: int = 30, 
                 use_existing: bool = False, enable_cache: bool = True):
        """Initialize RAG application"""
        
        self.website_url = website_url
        self.max_pages = max_pages
        self.enable_cache = enable_cache
        self.need_initialization = False
        
        rag_logger.info(f"Initializing RAGApp with website: {website_url}, max_pages: {max_pages}")
        
        self.cache = SemanticCache() if enable_cache else None
        if self.cache:
            stats = self.cache.get_cache_stats()
            rag_logger.info(f"Cache stats: {stats['total_entries']} entries")
        
        vector_store_path = "./data/website_chroma_db"
        
        if use_existing and os.path.exists(vector_store_path):
            rag_logger.info("Using existing vector store...")
            self.graph = build_rag_graph(self.cache)
            rag_logger.info("RAG system ready!")
            return
        
        # Don't scrape in constructor - will be done on demand
        rag_logger.info(f"RAG system will initialize on first request: {website_url}")
        self.need_initialization = True
        self.graph = None
    
    def initialize(self):
        """Initialize the RAG system by scraping and building vector store"""
        if hasattr(self, 'need_initialization') and self.need_initialization:
            rag_logger.info(f"Starting RAG initialization - scraping website: {self.website_url}")
            
            try:
                scraper = SPAScraper(self.website_url, self.max_pages, wait_time=5000)
                documents = scraper.scrape_sync()
                
                if not documents:
                    raise ValueError(f"No content scraped from {self.website_url}")
                
                rag_logger.info(f"Found {len(documents)} pages")
                
                chunks = chunk_documents(documents)
                rag_logger.info(f"Created {len(chunks)} chunks")
                
                self.vector_store = create_vector_store(chunks)
                rag_logger.info("Vector store created")
                
                self.graph = build_rag_graph(self.cache)
                rag_logger.info("RAG graph built")
                
                self.need_initialization = False
                rag_logger.info("RAG system initialization complete!")
                
            except Exception as e:
                rag_logger.error(f"RAG initialization failed: {e}", exc_info=True)
                raise
    
    def ask(self, question: str, thread_id: str = "1", stream: bool = False) -> Tuple[str, bool]:
        """Ask a question"""
        rag_logger.info(f"Processing question: {question[:50]}...")
        
        # Initialize if needed
        if self.need_initialization:
            rag_logger.info("Auto-initializing RAG system...")
            self.initialize()
        
        if self.graph is None:
            error_msg = "RAG system not properly initialized"
            rag_logger.error(error_msg)
            raise ValueError(error_msg)
        
        initial_state = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "needs_retrieval": True,
            "from_cache": False
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            if stream:
                rag_logger.info("Streaming mode enabled")
                state = retrieve_documents(initial_state)
                state = format_context(state)
                
                full_answer = ""
                for chunk in generate_answer_streaming(state, self.cache):
                    full_answer += chunk
                
                is_cached = "CACHED" in full_answer
                rag_logger.info(f"Streaming complete. Source: {'CACHE' if is_cached else 'GENERATED'}")
                return full_answer, is_cached
            else:
                result = self.graph.invoke(initial_state, config)
                is_cached = result.get("from_cache", False)
                rag_logger.info(f"Answer generated. Source: {'CACHE' if is_cached else 'GENERATED'}")
                return result["answer"], is_cached
        except Exception as e:
            rag_logger.error(f"Error processing question: {e}", exc_info=True)
            raise
    
    def get_cache_stats(self) -> Dict:
        if self.cache:
            return self.cache.get_cache_stats()
        return {"message": "Cache is disabled"}
    
    def clear_cache(self):
        if self.cache:
            self.cache.clear()
            rag_logger.info("Cache cleared by user request")

# ==================== INITIALIZE RAG APP ====================
# Initialize RAG with existing data if available
rag_app = None
try:
    rag_app = RAGApp(use_existing=True, enable_cache=True)
    rag_logger.info("RAG app initialized with existing data")
except Exception as e:
    rag_logger.warning(f"RAG initialization with existing data failed: {e}")
    rag_logger.info("Will initialize on first RAG request")

# ==================== FASTAPI ENDPOINTS ====================

# --- Health Endpoint ---
@app.get("/health")
def health():
    api_logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "RAG System API",
        "timestamp": datetime.now().isoformat()
    }

# --- Root Endpoint ---
@app.get("/")
def root():
    return {
        "service": "RAG System API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "rag_ask": "POST /rag/ask",
            "rag_init": "POST /rag/init",
            "rag_status": "GET /rag/status",
            "cache_stats": "GET /rag/cache/stats",
            "cache_clear": "POST /rag/cache/clear",
            "logs_recent": "GET /logs/recent",
            "logs_stream": "GET /logs/stream",
            "websocket_logs": "ws://host/ws/logs"
        }
    }

# ==================== LOG STREAMING ENDPOINTS ====================

@app.get("/logs/recent")
def get_recent_logs(limit: int = 100, level: Optional[str] = None):
    """Get recent log entries"""
    api_logger.info(f"Fetching recent logs (limit: {limit}, level: {level})")
    
    try:
        log_file = "logs/app.log"
        if not os.path.exists(log_file):
            return {"logs": [], "message": "No log file found"}
        
        logs = []
        with open(log_file, 'r') as f:
            # Read last N lines
            lines = f.readlines()
            lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in lines:
                try:
                    # Parse log line
                    parts = line.strip().split(' - ', 3)
                    if len(parts) >= 3:
                        timestamp = parts[0]
                        level_name = parts[1]
                        message = parts[3] if len(parts) > 3 else parts[2]
                        
                        # Filter by level if specified
                        if level and level_name != level.upper():
                            continue
                        
                        logs.append({
                            'timestamp': timestamp,
                            'level': level_name,
                            'message': message
                        })
                except Exception as e:
                    # Skip malformed lines
                    continue
        
        api_logger.info(f"Returning {len(logs)} recent logs")
        return {"logs": logs, "total": len(logs)}
        
    except Exception as e:
        api_logger.error(f"Error fetching recent logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch logs")

@app.get("/logs/stream")
async def stream_logs():
    """Stream logs as Server-Sent Events"""
    api_logger.info("Log stream connection opened")
    
    async def generate():
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to log stream'})}\n\n"
        
        # Send some recent logs
        try:
            log_file = "logs/app.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Send last 10 lines
                    for line in lines[-10:]:
                        yield f"data: {json.dumps({'type': 'recent', 'log': line.strip()})}\n\n"
                        await asyncio.sleep(0.1)
        except Exception as e:
            api_logger.error(f"Error sending recent logs: {e}")
        
        # Continue streaming new logs
        last_position = 0
        while True:
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                        for line in new_lines:
                            # Send each new line as an event
                            yield f"data: {json.dumps({'type': 'new', 'log': line.strip()})}\n\n"
                            await asyncio.sleep(0.05)
                
                await asyncio.sleep(0.5)  # Check for new logs every 500ms
            except Exception as e:
                api_logger.error(f"Error streaming logs: {e}")
                await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

# ==================== WEBSOCKET LOG STREAM ====================

@app.websocket("/ws/logs")
async def websocket_log_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    ws_logger.info(f"WebSocket client connected: {websocket.client}")
    
    active_websockets.add(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to log stream",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send recent logs
        try:
            log_file = "logs/app.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_logs = lines[-20:]  # Send last 20 lines
                    for line in recent_logs:
                        try:
                            parts = line.strip().split(' - ', 3)
                            if len(parts) >= 3:
                                await websocket.send_json({
                                    "type": "recent",
                                    "timestamp": parts[0],
                                    "level": parts[1],
                                    "message": parts[3] if len(parts) > 3 else parts[2],
                                    "full_log": line.strip()
                                })
                        except Exception:
                            continue
                        await asyncio.sleep(0.05)
        except Exception as e:
            ws_logger.error(f"Error sending recent logs via WebSocket: {e}")
        
        # Send a separator
        await websocket.send_json({
            "type": "separator",
            "message": "=== Live Log Stream ===",
            "timestamp": datetime.now().isoformat()
        })
        
        # Set up a queue listener for new logs
        log_queue_listener = asyncio.Queue()
        
        # Create a thread to listen to the log queue
        def listen_to_queue():
            while True:
                try:
                    log_entry = log_queue.get(timeout=1)
                    asyncio.run_coroutine_threadsafe(
                        log_queue_listener.put(log_entry),
                        asyncio.get_event_loop()
                    )
                except queue.Empty:
                    continue
                except Exception as e:
                    ws_logger.error(f"Error in queue listener: {e}")
                    break
        
        # Start the listener thread
        listener_thread = threading.Thread(target=listen_to_queue, daemon=True)
        listener_thread.start()
        
        # Main loop - send logs to WebSocket
        while True:
            try:
                # Wait for new log entries
                log_entry = await asyncio.wait_for(
                    log_queue_listener.get(), 
                    timeout=1.0
                )
                
                # Send to WebSocket
                await websocket.send_json({
                    "type": "live",
                    **log_entry
                })
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                continue
            except WebSocketDisconnect:
                ws_logger.info(f"WebSocket client disconnected: {websocket.client}")
                break
            except Exception as e:
                ws_logger.error(f"Error in WebSocket loop: {e}")
                break
                
    except WebSocketDisconnect:
        ws_logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
        ws_logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Remove client from active connections
        active_websockets.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
        ws_logger.info(f"WebSocket client removed: {websocket.client}")

# ==================== RAG Endpoints ====================

class RAGQuestion(BaseModel):
    question: str
    stream: bool = False
    thread_id: str = "1"

class RAGResponse(BaseModel):
    answer: str
    from_cache: bool = False
    success: bool = True

@app.post("/rag/ask")
async def ask_rag(question_data: RAGQuestion):
    """Ask a question to the RAG system"""
    api_logger.info(f"RAG question received: {question_data.question[:50]}...")
    
    global rag_app
    
    if rag_app is None:
        api_logger.info("RAG app is None, creating new instance")
        try:
            rag_app = RAGApp(use_existing=True, enable_cache=True)
        except Exception as e:
            api_logger.error(f"Failed to create RAG app: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"RAG system not initialized. Please initialize first with /rag/init. Error: {str(e)}"
            )
    
    try:
        # Run the ask method in a thread pool to avoid blocking the event loop
        def ask_sync():
            return rag_app.ask(
                question_data.question, 
                thread_id=question_data.thread_id,
                stream=question_data.stream
            )
        
        loop = asyncio.get_event_loop()
        answer, from_cache = await loop.run_in_executor(None, ask_sync)
        api_logger.info(f"RAG question answered. Source: {'CACHE' if from_cache else 'GENERATED'}")
        return RAGResponse(answer=answer, from_cache=from_cache)
    except Exception as e:
        api_logger.error(f"Error in RAG ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/init")
async def init_rag(website_url: str = "https://bkumar0823.space/", max_pages: int = 30):
    """Initialize or reinitialize the RAG system with fresh scraping"""
    api_logger.info(f"Initializing RAG with website: {website_url}, max_pages: {max_pages}")
    
    global rag_app
    try:
        # Create new RAG app instance
        rag_app = RAGApp(website_url=website_url, max_pages=max_pages, use_existing=False, enable_cache=True)
        
        # Run initialization in a thread to avoid blocking the event loop
        def init_rag_sync():
            rag_app.initialize()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_rag_sync)
        
        api_logger.info("RAG system initialized successfully")
        return {
            "message": "RAG system initialized successfully",
            "website": website_url,
            "max_pages": max_pages
        }
    except Exception as e:
        api_logger.error(f"Failed to initialize RAG: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")

@app.get("/rag/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    api_logger.info("Cache stats requested")
    
    global rag_app
    if rag_app is None:
        api_logger.warning("Cache stats requested but RAG not initialized")
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        stats = rag_app.get_cache_stats()
        api_logger.debug(f"Cache stats: {stats}")
        return stats
    except Exception as e:
        api_logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get cache stats")

@app.post("/rag/cache/clear")
async def clear_cache():
    """Clear the RAG cache"""
    api_logger.info("Cache clear requested")
    
    global rag_app
    if rag_app is None:
        api_logger.warning("Cache clear requested but RAG not initialized")
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        rag_app.clear_cache()
        api_logger.info("Cache cleared successfully")
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        api_logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/rag/status")
async def rag_status():
    """Get RAG system status"""
    api_logger.debug("RAG status requested")
    
    global rag_app
    if rag_app is None:
        return {
            "status": "not_initialized",
            "message": "RAG system not initialized. Use /rag/init to initialize.",
            "endpoints": {
                "init": "/rag/init",
                "ask": "/rag/ask",
                "status": "/rag/status",
                "cache_stats": "/rag/cache/stats",
                "cache_clear": "/rag/cache/clear"
            }
        }
    
    try:
        status_info = {
            "status": "ready",
            "cache_enabled": rag_app.enable_cache,
            "cache_stats": rag_app.get_cache_stats() if rag_app.cache else None,
            "needs_initialization": rag_app.need_initialization if hasattr(rag_app, 'need_initialization') else False,
            "website_url": rag_app.website_url,
            "max_pages": rag_app.max_pages
        }
        
        if hasattr(rag_app, 'need_initialization') and rag_app.need_initialization:
            status_info["status"] = "initializing_needed"
            status_info["message"] = "RAG system needs initialization. It will auto-initialize on first query or use /rag/init"
        
        return status_info
    except Exception as e:
        api_logger.error(f"Error getting RAG status: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Run on startup - initialize RAG if needed"""
    logger.info("=== APPLICATION STARTING ===")
    logger.info(f"FastAPI version: {app.version}")
    logger.info(f"Title: {app.title}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    global rag_app
    if rag_app and hasattr(rag_app, 'need_initialization') and rag_app.need_initialization:
        logger.info("🔄 RAG system will initialize on first request...")
    
    logger.info("=== APPLICATION STARTUP COMPLETE ===")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    logger.info("=== APPLICATION SHUTTING DOWN ===")
    logger.info("Closing WebSocket connections...")
    for ws in active_websockets:
        try:
            await ws.close()
        except Exception:
            pass
    active_websockets.clear()
    
    logger.info("=== APPLICATION SHUTDOWN COMPLETE ===")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)