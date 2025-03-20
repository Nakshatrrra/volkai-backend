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
        end_token = "<|endoftext|>"
        
        try:
            for chunk in output:
                if isinstance(chunk, dict) and "token" in chunk:
                    token = chunk["token"]
                    
                    # Check if adding this token would create or complete the end token
                    test_text = accumulated_text + token
                    if end_token in test_text:
                        # Find where the end token starts in the accumulated text
                        for i in range(len(test_text) - len(end_token) + 1):
                            if test_text[i:i+len(end_token)] == end_token:
                                # If this token is part of the end token, don't send it
                                if i <= len(accumulated_text):
                                    logger.debug("End token starting, stopping stream")
                                    break
                        break  # Exit the loop completely
                    
                    accumulated_text += token
                    yield f"data: {token}\n\n"  # Format for proper event streaming
                elif isinstance(chunk, str):
                    # Similar check for string chunks
                    test_text = accumulated_text + chunk
                    if end_token in test_text:
                        break
                    
                    accumulated_text += chunk
                    yield f"data: {chunk}\n\n"
                
                await asyncio.sleep(0.01)  # Slightly longer sleep to ensure proper chunking
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"
        
        logger.debug("Streaming completed")

    return StreamingResponse(event_generator(), media_type="text/event-stream")