import os
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import asyncio

load_dotenv()  # Load environment variables

# Load Hugging Face API credentials
HF_ENDPOINT = os.getenv("HF_ENDPOINT")
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(base_url=HF_ENDPOINT, token=HF_TOKEN)


def format_prompt(messages):
    """Formats messages into a format the model understands."""
    prompt = "### Context: You're VolkAI, Created by Kairosoft AI Solutions Limited.\n\n"
    
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"### Human: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"### Assistant: {msg['content']}\n"
        elif msg["role"] == "system":
            prompt += f"### Context: {msg['content']}\n"
    
    prompt += "### Assistant:"
    return prompt


async def stream_response(messages, max_tokens=200, temperature=0.7):
    """Streams the response from the Hugging Face inference API in real-time."""
    prompt = format_prompt(messages)

    async def event_generator():
        output = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        try:
            for chunk in output:
                if isinstance(chunk, dict) and "token" in chunk:
                    yield f"data: {chunk['token']}\n\n"  # Format for proper event streaming
                elif isinstance(chunk, str):
                    yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)  # Allows event loop to send data immediately
        except Exception as e:
            print(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
