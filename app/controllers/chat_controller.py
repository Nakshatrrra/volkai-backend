import os
import logging
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    logger.debug(f"Generated prompt for streaming")

    async def event_generator():
        output = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        accumulated_text = ""  # Track the complete text
        buffer = ""  # Buffer to check for "<|endoftext|>"
        end_token = "<|endoftext|>"

        try:
            for chunk in output:
                if isinstance(chunk, dict) and "token" in chunk:
                    token = chunk["token"]
                elif isinstance(chunk, str):
                    token = chunk
                else:
                    continue  # Skip if the format is unexpected

                buffer += token  # Add token to buffer

                # If buffer starts forming "<|endoftext|>", stop streaming before sending "<"
                if buffer.startswith("<") and end_token.startswith(buffer):
                    if end_token == buffer:  # If full end token is formed, break
                        break
                    continue  # Don't send tokens yet, wait for full confirmation

                accumulated_text += buffer
                yield f"data: {buffer}\n\n"
                buffer = ""  # Reset buffer after sending

                await asyncio.sleep(0.01)  # Slightly longer sleep to ensure proper chunking
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"
        
        logger.debug("Streaming completed")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
