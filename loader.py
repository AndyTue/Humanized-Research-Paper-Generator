import os
from pypdf import PdfReader
import docx

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def extract_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
    return text

def process_documents(file_paths):
    """
    Takes a list of file paths.
    Extracts text from all supported documents.
    Returns a combined string or a list of documents.
    For RAG, it's often better to keep document context, but for now we'll return a dictionary mapping filename to text.
    """
    docs = {}
    if not file_paths:
        return docs
        
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            text = extract_text_from_docx(file_path)
        elif ext == ".txt":
            text = extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {ext} for file {file_path}")
            continue
            
        if text.strip():
            docs[os.path.basename(file_path)] = text
            
    return docs
