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
        buffer = ""  # Buffer to accumulate tokens
        buffer_size = 8  # Number of tokens to buffer before sending
        end_token = "<|endoftext|>"
        
        try:
            for chunk in output:
                if isinstance(chunk, dict) and "token" in chunk:
                    token = chunk["token"]
                    
                    # Check if this token is part of the end token
                    test_text = accumulated_text + buffer + token
                    if end_token in test_text:
                        end_index = test_text.find(end_token)
                        # If there's content before the end token, send it
                        if end_index > len(accumulated_text):
                            content_to_send = test_text[len(accumulated_text):end_index]
                            if content_to_send:
                                yield f"data: {content_to_send}\n\n"
                        break  # Exit the loop completely
                    
                    accumulated_text += token
                    buffer += token
                    
                    # Send buffer when it reaches the desired size
                    if len(buffer) >= buffer_size:
                        yield f"data: {buffer}\n\n"  # Format for proper event streaming
                        buffer = ""  # Reset buffer after sending
                        
                elif isinstance(chunk, str):
                    # Similar check for string chunks
                    test_text = accumulated_text + buffer + chunk
                    if end_token in test_text:
                        end_index = test_text.find(end_token)
                        # If there's content before the end token, send it
                        if end_index > len(accumulated_text + buffer):
                            content_to_send = test_text[len(accumulated_text + buffer):end_index]
                            if content_to_send:
                                yield f"data: {content_to_send}\n\n"
                        break
                    
                    accumulated_text += chunk
                    buffer += chunk
                    
                    # Send buffer when it reaches the desired size
                    if len(buffer) >= buffer_size:
                        yield f"data: {buffer}\n\n"
                        buffer = ""  # Reset buffer after sending
                
                await asyncio.sleep(0.01)  # Slightly longer sleep to ensure proper chunking
            
            # Send any remaining buffered content at the end
            if buffer and not end_token in buffer:
                yield f"data: {buffer}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"
        
        logger.debug("Streaming completed")

    return StreamingResponse(event_generator(), media_type="text/event-stream")