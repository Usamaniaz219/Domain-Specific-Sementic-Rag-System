import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# from ..config.settings import settings
# from ..config.constants import FileType
# from ..utils.logger import get_logger

from config.settings import settings
from config.constants import FileType
from utils.logger import get_logger


logger = get_logger(__name__)

class DocumentProcessor:
    """Handles document loading, parsing, and chunking"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._load_text_file,
            '.pdf': self._load_pdf_file,
            '.docx': self._load_docx_file,
            '.html': self._load_html_file,
            '.md': self._load_markdown_file,
        }
        logger.info("Document processor initialized")
    
    def load_documents(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Load documents from various file formats"""
        documents = []
        
        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist, skipping")
                continue
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > settings.MAX_FILE_SIZE:
                logger.warning(f"File {file_path} is too large ({file_size} bytes), skipping")
                continue
                
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {file_extension}")
                continue
                
            try:
                loader = self.supported_extensions[file_extension]
                content = loader(file_path)
                
                documents.append({
                    'content': content,
                    'metadata': {
                        'source': str(file_path),
                        'file_type': FileType(file_extension[1:]),
                        'file_size': file_size,
                        'load_time': datetime.now().isoformat(),
                        'file_hash': self._calculate_file_hash(file_path)
                    }
                })
                logger.info(f"Successfully loaded {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks with overlap using semantic boundaries"""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            
            # Simple sentence-based splitting (in production, use more sophisticated methods)
            sentences = self._split_into_sentences(content)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > settings.CHUNK_SIZE and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = self._generate_chunk_id(doc['metadata']['source'], chunk_text)
                    
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'metadata': {
                            **doc['metadata'],
                            'chunk_length': current_length,
                            'chunk_hash': hashlib.md5(chunk_text.encode()).hexdigest()
                        }
                    })
                    
                    # Keep overlap for next chunk
                    overlap_size = int(settings.CHUNK_OVERLAP * 0.01 * len(current_chunk))
                    current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                    current_length = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(doc['metadata']['source'], chunk_text)
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_length': current_length,
                        'chunk_hash': hashlib.md5(chunk_text.encode()).hexdigest()
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load text file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load PDF file content using PyPDF2"""
        try:
            import PyPDF2
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except ImportError:
            logger.error("PyPDF2 is required for PDF processing")
            raise
    
    def _load_docx_file(self, file_path: Path) -> str:
        """Load DOCX file content using python-docx"""
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            logger.error("python-docx is required for DOCX processing")
            raise
    
    def _load_html_file(self, file_path: Path) -> str:
        """Load HTML file content and extract text"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except ImportError:
            logger.error("BeautifulSoup is required for HTML processing")
            raise
    
    def _load_markdown_file(self, file_path: Path) -> str:
        """Load Markdown file content"""
        return self._load_text_file(file_path)  # Same as text for now
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting (in production, use nltk or spaCy)"""
        import re
        # Basic sentence splitting by punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_chunk_id(self, source: str, text: str) -> str:
        """Generate unique ID for a chunk"""
        return hashlib.md5(f"{source}_{text}".encode()).hexdigest()
    