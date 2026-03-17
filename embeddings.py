from sentence_transformers import SentenceTransformer

class EmbeddingsModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text):
        """
        Embeds a single string of text.
        """
        if not text:
            return None
        return self.model.encode(text)
        
    def embed_texts(self, texts):
        """
        Embeds a list of strings.
        Returns a numpy array of embeddings.
        """
        if not texts:
            return []
        return self.model.encode(texts)
