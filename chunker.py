from token_utils import count_tokens

# ── Token-based chunk sizing constants ──────────────────────────────────────
CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 25
# Approximate characters-per-token ratio for break-point search
_CHARS_PER_TOKEN = 4


def chunk_text(text, chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS):
    """
    Splits text into chunks of approximately *chunk_size* tokens with
    *chunk_overlap* tokens of overlap.  Break-point search (newlines,
    periods, spaces) is still character-based for simplicity; only the
    size measurement uses real token counts.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    # Convert token targets to approximate character positions
    char_step = chunk_size * _CHARS_PER_TOKEN
    char_overlap = chunk_overlap * _CHARS_PER_TOKEN

    while start < text_length:
        end = start + char_step

        # If we're not at the end of the text, try to find a nice breaking point
        if end < text_length:
            # Look for a newline character to break at
            newline_pos = text.rfind('\n', start, end)
            if newline_pos != -1 and newline_pos > start + char_step // 2:
                end = newline_pos + 1
            else:
                # Look for a period to break at
                period_pos = text.rfind('. ', start, end)
                if period_pos != -1 and period_pos > start + char_step // 2:
                    end = period_pos + 2
                else:
                    # Look for a space to break at
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + char_step // 2:
                        end = space_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - char_overlap

        # Prevent infinite loops if overlap is too large
        if start <= end - char_step:
            start = end

    return chunks

def process_chunks(docs_dict, chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS):
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
