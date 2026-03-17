def chunk_text(text, chunk_size=800, chunk_overlap=100):
    """
    Splits text into chunks of approximately chunk_size characters with chunk_overlap.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a nice breaking point
        if end < text_length:
            # Look for a newline character to break at
            newline_pos = text.rfind('\n', start, end)
            if newline_pos != -1 and newline_pos > start + chunk_size // 2:
                end = newline_pos + 1
            else:
                # Look for a period to break at
                period_pos = text.rfind('. ', start, end)
                if period_pos != -1 and period_pos > start + chunk_size // 2:
                    end = period_pos + 2
                else:
                    # Look for a space to break at
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + chunk_size // 2:
                        end = space_pos + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - chunk_overlap
        
        # Prevent infinite loops if overlap is too large
        if start <= end - chunk_size:
            start = end
            
    return chunks

def process_chunks(docs_dict, chunk_size=800, chunk_overlap=100):
    """
    Takes a dictionary {filename: text}.
    Returns a list of dictionaries, each containing 'text' and 'source_file'.
    """
    all_chunks = []
    for filename, text in docs_dict.items():
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })
    return all_chunks
