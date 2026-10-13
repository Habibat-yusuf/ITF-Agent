
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from RAG import process_and_store_document
from RESE import research_itf
from WRY import write_itf_answer
from SHIELD import check_input, check_output
from FEED import process_feedback


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(
    title="ITF Multi-Agent RAG System",
    description=(
        "A role-based ITF information assistant "
        "powered by RAG and four specialized agents."
    ),
    version="1.0.0",
)

frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = [
    origin.strip()
    for origin in frontend_url.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 2. REQUEST MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    # Optional for compatibility with the earlier API.
    agent_type: str | None = None


class FeedbackRequest(BaseModel):
    rating: int
    comment: str = Field(default="", max_length=2000)
    question: str = Field(default="", max_length=2000)


# --------------------------------------------------
# 3. HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "application": "ITF Multi-Agent RAG System",
        "status": "running",
        "agents": ["Shield", "Rese", "Wry", "Feed"],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# --------------------------------------------------
# 4. MAIN MULTI-AGENT WORKFLOW
# --------------------------------------------------

@app.post("/api/agent/chat")
def agent_chat(request: QueryRequest):
    query = request.query.strip()

    # Agent 1: Shield checks the user's input.
    input_check = check_input(query)

    if not input_check["allowed"]:
        return {
            "status": "rejected",
            "agent": "Shield",
            "response": input_check["message"],
            "sources_used": [],
            "reason": input_check["reason"],
        }

    # Agent 2: Rese searches the RAG knowledge base.
    research = research_itf(query)

    # Agent 3: Wry prepares an answer from the evidence.
    writing = write_itf_answer(query, research)

    # Agent 4's workflow checkpoint: Shield checks
    # whether research succeeded before releasing an answer.
    output_check = check_output(
        writing.get("answer", ""),
        research,
    )

    if not output_check["approved"]:
        return {
            "status": "no_evidence",
            "agent": "Shield",
            "response": output_check["answer"],
            "sources_used": [],
            "reason": output_check["reason"],
        }

    return {
        "status": "success",
        "agent": "Wry",
        "response": output_check["answer"],
        "sources_used": writing.get("sources", []),
        "workflow": [
            "Shield",
            "Rese",
            "Wry",
            "Shield",
        ],
    }


# --------------------------------------------------
# 5. DOCUMENT UPLOAD AND RAG INGESTION
# --------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a document to upload.",
        )

    original_name = Path(file.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats are PDF, DOCX, and TXT.",
        )

    # Read at most 10 MB plus one byte to detect oversized files.
    content = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The document exceeds the 10 MB upload limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty.",
        )

    # Use a unique stored filename to avoid accidental overwrites.
    stored_name = f"{uuid.uuid4().hex}_{original_name}"
    saved_path = UPLOAD_DIR / stored_name

    try:
        saved_path.write_bytes(content)

        result = process_and_store_document(
            str(saved_path),
            source_name=original_name,
        )

        return {
            "status": "success",
            "message": (
                "Document processed and added to the RAG database."
            ),
            "filename": original_name,
            "chunks_stored": result["chunks_stored"],
            "extracted_text_file": result["text_file"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Document ingestion failed: {error}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed. Check the server logs, "
                "API credentials, and Supabase configuration."
            ),
        ) from error

    finally:
        await file.close()


# --------------------------------------------------
# 6. FEEDBACK ENDPOINT
# --------------------------------------------------

@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest):
    result = process_feedback(
        rating=request.rating,
        comment=request.comment,
        question=request.question,
    )

    if result.get("status") != "success":
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "Invalid feedback."),
        )

    return result


# --------------------------------------------------
# 7. LOCAL DEVELOPMENT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
