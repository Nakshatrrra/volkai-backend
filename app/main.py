from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat

app = FastAPI(title="VolkAI Chat API")

# Allow CORS for all origins (frontend can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat.router, prefix="/api")

@app.get("/")
def home():
    return {"message": "VolkAI Chat API is running!"}
