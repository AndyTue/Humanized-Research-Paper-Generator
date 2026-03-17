import os
from loader import process_documents
from chunker import process_chunks
from embeddings import EmbeddingsModel
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator
from humanizer import Humanizer
from dotenv import load_dotenv

load_dotenv()

class ResearchAgent:
    def __init__(self):
        self.embeddings_model = EmbeddingsModel()
        self.vector_store = VectorStore()
        # Initialize lazily to avoid immediate failure if API key is not set
        self.retriever = Retriever(self.vector_store, self.embeddings_model)
        self.generator = None
        self.humanizer = None
        self.is_processed = False
        
    def initialize_apis(self):
        # We initialize these here so we can catch missing API keys only when needed
        if self.generator is None:
            self.generator = Generator()
        if self.humanizer is None:
            self.humanizer = Humanizer()
            
    def process_files(self, file_paths):
        """
        Pipeline: Ingest -> Chunk -> Embed -> Store.
        """
        try:
            print("1. Extracting text from documents...")
            docs_dict = process_documents(file_paths)
            if not docs_dict:
                return "No valid text extracted from the provided files."
                
            print("2. Chunking text...")
            chunks = process_chunks(docs_dict)
            if not chunks:
                return "Error creating text chunks."
                
            print(f"3. Embedding {len(chunks)} chunks...")
            texts_to_embed = [c["text"] for c in chunks]
            embeddings = self.embeddings_model.embed_texts(texts_to_embed)
            
            print("4. Storing in FAISS...")
            # Re-initialize vector store for each new processing to clear previous state
            self.vector_store = VectorStore()
            self.retriever = Retriever(self.vector_store, self.embeddings_model)
            
            self.vector_store.add_chunks(chunks, embeddings)
            self.is_processed = True
            
            return f"Successfully processed {len(file_paths)} files. Extracted and indexed {len(chunks)} text chunks."
        except Exception as e:
            return f"Error during document processing: {e}"
            
    def generate_research_paper(self, query):
        """
        Pipeline: Retrieve (FAISS + Wiki) -> Generate -> Validate (happens in Generate) -> Humanize.
        """
        try:
            self.initialize_apis()
            
            # If no files were provided or processed, we just proceed with empty FAISS.
            # Retriever handles empty FAISS gracefully.
            
            print("1. Retrieving context (Local + Wikipedia)...")
            context = self.retriever.retrieve(query, top_k=6, wiki_sentences=15)
            
            print("2. Generating Paper draft...")
            draft_paper = self.generator.generate_ieee_paper(query, context)
            
            print("3. Humanizing text...")
            humanized_paper = self.humanizer.humanize_text(draft_paper)
            
            return humanized_paper
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating paper: {e}"
