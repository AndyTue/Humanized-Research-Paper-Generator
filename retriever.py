import wikipedia
from vector_store import VectorStore
from embeddings import EmbeddingsModel

class Retriever:
    def __init__(self, vector_store: VectorStore, embeddings_model: EmbeddingsModel):
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        
    def get_faiss_context(self, query, top_k=5):
        """
        Retrieves context chunks from FAISS.
        """
        query_emb = self.embeddings_model.embed_text(query)
        if query_emb is None:
            return ""
            
        results = self.vector_store.search(query_emb, k=top_k)
        
        context_parts = []
        for res in results:
            chunk = res["chunk"]
            text = chunk.get("text", "")
            source = chunk.get("source", "Unknown")
            context_parts.append(f"[Source: {source}]\n{text}")
            
        return "\n\n".join(context_parts)
        
    def get_wikipedia_context(self, query, sentences=10):
        """
        Searches Wikipedia for the query and returns context.
        """
        context = ""
        try:
            # Search for the best matching page title
            search_results = wikipedia.search(query, results=1)
            if search_results:
                page_title = search_results[0]
                page = wikipedia.page(page_title, auto_suggest=False)
                # Get a summary or full text. We use summary with high sentences for better context.
                # If page is too large, we might want to just get the summary or truncate to 2000 chars
                summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
                context = f"[Source: Wikipedia - {page_title}]\n{summary}"
            else:
                context = f"[Source: Wikipedia]\nNo matching articles found for query: {query}"
        except wikipedia.exceptions.DisambiguationError as e:
            # If multiple meanings, pick the first one
            try:
                page_title = e.options[0]
                summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
                context = f"[Source: Wikipedia - {page_title}]\n{summary}"
            except:
                context = f"[Source: Wikipedia]\nError resolving disambiguation for query: {query}"
        except Exception as e:
            context = f"[Source: Wikipedia]\nError retrieving data: {e}"
            
        return context
        
    def retrieve(self, query, top_k=5, wiki_sentences=10):
        """
        Retrieves formatted context from both FAISS and Wikipedia.
        """
        faiss_context = self.get_faiss_context(query, top_k)
        wiki_context = self.get_wikipedia_context(query, wiki_sentences)
        
        combined_context = f"=== USER DOCUMENTS CONTEXT ===\n{faiss_context}\n\n"
        combined_context += f"=== WIKIPEDIA CONTEXT ===\n{wiki_context}"
        
        return combined_context
