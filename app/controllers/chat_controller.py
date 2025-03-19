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


async def stream_response(messages, max_tokens=500, temperature=0.5):
    """Streams the response from the Hugging Face inference API in real-time."""
    prompt = format_prompt(messages)

    async def event_generator():
        output = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        last_tokens = []  # Store last few tokens to check for end-of-text

        try:
            for chunk in output:
                if isinstance(chunk, dict) and "token" in chunk:
                    token = chunk["token"]
                    last_tokens.append(token)

                    # Keep only the last 10 tokens to check for "<|endoftext|>"
                    if len(last_tokens) > 10:
                        last_tokens.pop(0)

                    last_text = "".join(last_tokens)
                    if "<|endoftext|>" in last_text:
                        break  # Stop processing when end token is detected

                    yield f"data: {token}\n\n"  # Format for proper event streaming
                elif isinstance(chunk, str):
                    yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)  # Allows event loop to send data immediately
        except Exception as e:
            print(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")