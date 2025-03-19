from fastapi import APIRouter, HTTPException
from app.controllers.chat_controller import stream_response

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: dict):
    """
    Chat endpoint that takes input messages and returns a streamed response.
    """
    try:
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 200)
        temperature = request.get("temperature", 0.7)
        
        if not messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        return await stream_response(messages, max_tokens, temperature)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
