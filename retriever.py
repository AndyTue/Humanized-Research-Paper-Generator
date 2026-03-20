import wikipedia
from vector_store import VectorStore
from embeddings import EmbeddingsModel
from token_utils import count_tokens, truncate_to_tokens

# ── Token budget constants ──────────────────────────────────────────────────
CONTEXT_TOKEN_BUDGET = 23500   # Total budget for combined context
FAISS_BUDGET_RATIO = 0.6       # 60 % of budget for FAISS chunks
WIKI_BUDGET_RATIO = 0.4        # 40 % of budget for Wikipedia

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
                # Get a summary (the old wikipedia.page() call was removed —
                # it downloaded the full article but was never used).
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
        Enforces a hard token budget split 60/40 between FAISS and Wikipedia.
        """
        faiss_context = self.get_faiss_context(query, top_k)
        wiki_context = self.get_wikipedia_context(query, wiki_sentences)

        # ── Token budgeting ─────────────────────────────────────────────
        faiss_budget = int(CONTEXT_TOKEN_BUDGET * FAISS_BUDGET_RATIO)
        wiki_budget = int(CONTEXT_TOKEN_BUDGET * WIKI_BUDGET_RATIO)

        if count_tokens(faiss_context) > faiss_budget:
            faiss_context = truncate_to_tokens(faiss_context, faiss_budget)
        if count_tokens(wiki_context) > wiki_budget:
            wiki_context = truncate_to_tokens(wiki_context, wiki_budget)
        
        combined_context = f"=== USER DOCUMENTS CONTEXT ===\n{faiss_context}\n\n"
        combined_context += f"=== WIKIPEDIA CONTEXT ===\n{wiki_context}"
        
        return combined_context
