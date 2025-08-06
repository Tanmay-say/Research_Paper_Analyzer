import os
import logging
import pickle
from datetime import datetime
from config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from werkzeug.utils import secure_filename
import hashlib
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalMemory:
    """Local memory system for storing processed documents and metadata."""
    
    def __init__(self, memory_file="local_memory.pkl"):
        self.memory_file = memory_file
        self.documents = {}  # filename -> document metadata
        self.chunks = {}     # filename -> list of text chunks
        self.load_memory()
    
    def load_memory(self):
        """Load memory from disk."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.chunks = data.get('chunks', {})
                logger.info(f"Loaded {len(self.documents)} documents from local memory")
        except Exception as e:
            logger.warning(f"Could not load memory: {e}")
    
    def save_memory(self):
        """Save memory to disk."""
        try:
            data = {
                'documents': self.documents,
                'chunks': self.chunks,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Memory saved to disk")
        except Exception as e:
            logger.error(f"Could not save memory: {e}")
    
    def add_document(self, filename: str, metadata: Dict[str, Any], chunks: List[str]):
        """Add a document to memory."""
        self.documents[filename] = metadata
        self.chunks[filename] = chunks
        self.save_memory()
    
    def get_document(self, filename: str) -> Dict[str, Any]:
        """Get document metadata."""
        return self.documents.get(filename, {})
    
    def get_chunks(self, filename: str) -> List[str]:
        """Get document chunks."""
        return self.chunks.get(filename, [])
    
    def list_documents(self) -> List[str]:
        """List all stored documents."""
        return list(self.documents.keys())
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search through all chunks for relevant content."""
        results = []
        for filename, chunks in self.chunks.items():
            for i, chunk in enumerate(chunks):
                if query.lower() in chunk.lower():
                    results.append({
                        'filename': filename,
                        'chunk_index': i,
                        'content': chunk,
                        'metadata': self.documents.get(filename, {})
                    })
        # Sort by relevance (simple keyword matching for now)
        results.sort(key=lambda x: x['content'].lower().count(query.lower()), reverse=True)
        return results[:top_k]

# Global memory instance
local_memory = LocalMemory()

class GroqEmbeddings:
    """Custom embedding class using Groq for text embeddings via text generation."""
    
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.dimension = 768  # Standard embedding dimension
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        """Generate a simple hash-based embedding for text."""
        # Create a simple but consistent embedding using text hashing
        # This is a simplified approach - in production you'd use proper embeddings
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to vector
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension:
            embedding.extend(embedding[:self.dimension - len(embedding)])
        
        return embedding[:self.dimension]

def validate_file(file):
    """Validate the uploaded file.
    
    Checks if the file has a name and if it is a PDF.
    Returns the filename and an error message if validation fails.
    """
    filename = file.filename
    if not filename:
        return None, "File has no name."
    if not filename.endswith('.pdf'):
        return filename, f"File '{filename}' is not a PDF."
    return filename, None

def process_file(file, vectorstore):
    """Process a single uploaded file with enhanced local memory.
    
    Saves the file, processes it to extract text, stores in local memory,
    and ingests the text into the vectorstore.
    Returns the filename and an error message if processing fails.
    """
    data_directory = 'data'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    filename = secure_filename(file.filename)
    file_path = os.path.join('data', filename)
    file.save(file_path)
    logger.info(f"File saved: {file_path}")

    try:
        # Process the PDF and extract text chunks
        texts = process_pdf(file_path)
        
        # Store in local memory for fast retrieval
        metadata = {
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path),
            'chunk_count': len(texts),
            'source': 'uploaded_pdf'
        }
        
        # Extract text content from documents
        chunks = [doc.page_content for doc in texts]
        
        # Add to local memory
        local_memory.add_document(filename, metadata, chunks)
        logger.info(f"Added {filename} to local memory with {len(chunks)} chunks")
        
        # Also add to vectorstore for semantic search
        ingest_to_vectorstore(texts, vectorstore)
        logger.info(f"File successfully processed and added to database: {filename}")
        
        os.remove(file_path)  # Clean up the temporary file
        return filename, None
    except Exception as e:
        error_msg = f"Error processing file '{filename}': {str(e)}"
        logger.error(error_msg)
        return filename, error_msg

def get_vectorstore(embeddings):
    """Get or create a ChromaDB vectorstore.
    
    Initializes ChromaDB with persistent storage and returns the vectorstore.
    """
    try:
        # Create persistent directory for ChromaDB
        persist_directory = "./chroma_db"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            logger.info(f"Created ChromaDB directory: {persist_directory}")
        
        # Initialize ChromaDB with persistent storage
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="research_papers"
        )
        
        logger.info("ChromaDB vectorstore initialized successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")
        raise

def split_documents(documents):
    """Split documents into text chunks.
    
    Uses a recursive text splitter to divide documents into manageable chunks
    for processing and ingestion. Optimized for research papers.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for research papers
        chunk_overlap=200,  # More overlap for better context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separation for academic content
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Created {len(texts)} chunks from documents")
    return texts

def process_pdf(pdf_path):
    """Process a single PDF file and return text chunks with enhanced error handling.
    
    Loads the PDF, extracts text, and splits it into chunks.
    Raises an error if the file path is invalid.
    """
    try:
        # Ensure the path is a valid file
        if not os.path.isfile(pdf_path):
            raise ValueError(f"Provided path '{pdf_path}' is not a valid file.")

        # Load and process the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No text content found in the PDF file.")
        
        logger.info(f"Loaded {len(documents)} document(s) from file: {pdf_path}")
        
        # Split documents into chunks
        chunks = split_documents(documents)
        
        # Validate chunks
        if not chunks:
            raise ValueError("No text chunks generated from the PDF.")
        
        logger.info(f"Generated {len(chunks)} text chunks from PDF")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF file '{pdf_path}': {str(e)}")
        raise

def get_pdf_summary(filename: str) -> str:
    """Get a summary of a specific PDF from local memory."""
    try:
        metadata = local_memory.get_document(filename)
        chunks = local_memory.get_chunks(filename)
        
        if not chunks:
            return f"No content found for {filename}"
        
        # Create a simple summary from the first few chunks
        summary_chunks = chunks[:3]  # First 3 chunks
        summary = "\n\n".join(summary_chunks)
        
        if len(summary) > 1000:
            summary = summary[:1000] + "..."
        
        return f"Summary of {filename}:\n{summary}"
        
    except Exception as e:
        logger.error(f"Error getting summary for {filename}: {e}")
        return f"Error retrieving summary for {filename}"

def search_pdf_content(query: str, filename: str = None) -> List[Dict[str, Any]]:
    """Search for content in PDFs using local memory."""
    try:
        if filename:
            # Search in specific file
            chunks = local_memory.get_chunks(filename)
            results = []
            for i, chunk in enumerate(chunks):
                if query.lower() in chunk.lower():
                    results.append({
                        'filename': filename,
                        'chunk_index': i,
                        'content': chunk,
                        'metadata': local_memory.get_document(filename)
                    })
            return results
        else:
            # Search in all files
            return local_memory.search_chunks(query)
            
    except Exception as e:
        logger.error(f"Error searching PDF content: {e}")
        return []

def ingest_to_vectorstore(texts, vectorstore):
    """Add documents to vectorstore.
    
    Ingests the provided text chunks into the specified vectorstore
    and logs the result.
    """
    try:
        logger.info(f"Starting ingestion of {len(texts)} documents to the vectorstore.")
        # Ingest documents into the vectorstore
        result = vectorstore.add_documents(texts)
        logger.info(f"Successfully ingested {len(texts)} documents to the vectorstore.")
        return result
    except Exception as e:
        logger.error(f"Error ingesting documents to vectorstore: {str(e)}")
        raise