import os
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client
from google.genai import types

# Load environment variables from .env
load_dotenv()

print("Starting Script...")# Explicitly find and load the .env file in the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Verify keys are loaded (optional check)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not GEMINI_API_KEY:
     raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")


# Initialize Gemini and Supabase clients
client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_text_from_file(file_path: str) -> str:
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    elif ext in [".docx", ".doc"]:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        
    return chunks

def get_embedding(text: str) -> list[float]:
    """Generates vector embeddings using Google gemini-embedding-2-preview."""
    response = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=text.replace("\n", " "),
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    return response.embeddings[0].values

def process_and_store_document(file_path: str, source_name: str):
    print(f"Processing {file_path}...")
    raw_text = extract_text_from_file(file_path)
    chunks = chunk_text(raw_text)
    
    print(f"Generated {len(chunks)} chunks. Storing in Supabase...")
    for chunk in chunks:
        # Clean null bytes and invalid unicode characters that break PostgreSQL
        clean_chunk = chunk.replace("\x00", "").strip()
        if not clean_chunk:
            continue

        embedding = get_embedding(clean_chunk)
        supabase.table("itf_documents").insert({
            "content": clean_chunk,
            "source": source_name,
            "embedding": embedding
        }).execute()
        
    print(f"Successfully ingested {source_name} into database!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "ITF BROCHURE.pdf")
    
    process_and_store_document(file_path, "ITF 2022 Brochure")