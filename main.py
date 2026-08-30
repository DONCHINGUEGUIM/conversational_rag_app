import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Imports for Qdrant and Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="Conversational RAG API with Groq & Qdrant", version="1.0")

# 1. Initialize Clients
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
embedding_model = TextEmbedding()  # Uses lightweight BAAI/bge-small-en-v1.5 by default

COLLECTION_NAME = "educational_docs"

# Ensure Qdrant collection exists on startup
def init_qdrant():
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# Warm up embedding model on boot so the first chat request isn't delayed
@app.on_event("startup")
async def startup_event():
    init_qdrant()
    # Trigger a dummy embedding to force model download/loading into RAM on boot
    list(embedding_model.embed(["warmup"]))
    print(" FastEmbed model loaded and warm!")

    
# 2. Chat setup
llm = ChatGroq(model_name="llama-3.1-8b-instant")
history_db = {}

def get_session_history(session_id: str):
    if session_id not in history_db:
        history_db[session_id] = ChatMessageHistory()
    return history_db[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an educational assistant. Use the provided context to help answer user questions clearly."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}")
])

chain = prompt | llm
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history"
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Search Qdrant for relevant context
    query_vector = list(embedding_model.embed([request.message]))[0].tolist()
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    ).points

    context = "\n".join([hit.payload["text"] for hit in search_results if "text" in hit.payload])
    
    # Combine user message with vector context if present
    full_message = f"Context:\n{context}\n\nUser Question: {request.message}" if context else request.message

    res = conversational_chain.invoke(
        {"message": full_message},
        config={"configurable": {"session_id": request.session_id}}
    )
    return res.content

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Extracts text from uploaded PDF, creates embeddings, and stores in Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    content = await file.read()
    reader = PdfReader(BytesIO(content))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    # Chunk text (simple 500-character chunking)
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - 50)]

    # Generate embeddings and store points in Qdrant
    embeddings = list(embedding_model.embed(chunks))
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=hash(f"{file.filename}_{idx}") & 0x7FFFFFFF,
                vector=vector.tolist(),
                payload={"filename": file.filename, "text": chunk}
            )
        )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"message": f"Successfully indexed {len(chunks)} chunks from {file.filename}"}