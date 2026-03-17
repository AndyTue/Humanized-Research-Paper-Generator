import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = [] # Stores {"text": ..., "source": ...}
        
    def add_chunks(self, chunks, embeddings):
        """
        Adds chunks and their embeddings to the FAISS index.
        chunks: list of dicts {"text": str, "source": str}
        embeddings: numpy array of shape (N, D)
        """
        if len(chunks) == 0 or len(embeddings) == 0:
            return
            
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        
    def search(self, query_embedding, k=5):
        """
        Searches the index for the top k closest vectors to the query.
        Returns a list of matching chunks.
        """
        if self.index.ntotal == 0:
            return []
            
        query_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": distances[0][i]
                })
        return results
        
    def save(self, filepath_prefix):
        """
        Saves the FAISS index and chunk metadata.
        """
        faiss.write_index(self.index, f"{filepath_prefix}.index")
        with open(f"{filepath_prefix}_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
            
    def load(self, filepath_prefix):
        """
        Loads the FAISS index and chunk metadata.
        """
        if os.path.exists(f"{filepath_prefix}.index"):
            self.index = faiss.read_index(f"{filepath_prefix}.index")
        if os.path.exists(f"{filepath_prefix}_chunks.pkl"):
            with open(f"{filepath_prefix}_chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
