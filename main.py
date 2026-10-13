import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

# Explicitly load environment variables
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")

app = FastAPI(title="ITF Multi-Agent RAG System (Gemini)", version="1.0")

client = genai.Client(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

REFUSAL_MESSAGE = "I don't have that information. Please contact ITF directly."

class QueryRequest(BaseModel):
    agent_type: str  # "services" or "reimbursement"
    query: str

@app.post("/api/agent/chat")
def agent_chat(req: QueryRequest):
    # 1. Generate query embedding using the current embedding model
    emb_resp = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=req.query.replace("\n", " "),
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    query_embedding = emb_resp.embeddings[0].values
    
    # 2. Retrieve context via Supabase RPC vector search
    response = supabase.rpc("match_itf_documents", {
        "query_embedding": query_embedding,
        "match_threshold": 0.3,
        "match_count": 3
    }).execute()
    
    data = response.data
    context_text = "\n\n".join([row['content'] for row in data]) if data else ""
    
    # 3. Define agent scope and guardrails
    if req.agent_type == "services":
        system_prompt = (
            "You are the official ITF Services Agent. Your sole scope is answering questions regarding the Industrial Training Fund (ITF) "
            "mandate, vision, mission, core values, client eligibility criteria, training programmes, and statutory services (such as SIWES). "
            "CRITICAL GUARDRAIL: If the user asks about topics outside your scope, you MUST reply precisely with: "
            f"'{REFUSAL_MESSAGE}' Do not guess or extrapolate."
        )
    elif req.agent_type == "reimbursement":
        system_prompt = (
            "You are the official ITF Reimbursement Agent. Your sole scope is answering questions regarding the ITF Training Contribution Act (1% payroll levy) "
            "and the ITF Reimbursement Scheme. "
            "CRITICAL GUARDRAIL: If the user asks about unrelated topics or information missing from the retrieved context, "
            f"you MUST reply precisely with: '{REFUSAL_MESSAGE}' Do not hallucinate."
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid agent type specified.")

    prompt = f"{system_prompt}\n\nContext Information:\n{context_text}\n\nUser Question: {req.query}"
    
    # 4. Generate response using the current Flash model
    ai_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    
    answer = ai_response.text.strip()
    sources = list(set([row['source'] for row in data])) if data else []

    return {
        "agent": req.agent_type,
        "response": answer,
        "sources_used": sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)