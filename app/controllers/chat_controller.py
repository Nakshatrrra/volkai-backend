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
            stream=True,
            decoder_input_details=True
        )

        accumulated_text = ""  # Track the complete text
        end_token = "<|endoftext|>"
        buffer = ""  # Buffer to help detect end token across chunks
        
        try:
            for chunk in output:
                token = ""
                if isinstance(chunk, dict) and "token" in chunk:
                    token = chunk["token"]
                elif isinstance(chunk, str):
                    token = chunk
                
                if not token:
                    continue
                
                # Add to buffer for end token detection
                buffer += token
                
                # If buffer contains more characters than end token, we can send some
                if len(buffer) > len(end_token):
                    # Check if end token is in buffer
                    if end_token in buffer:
                        # Get position of end token
                        end_pos = buffer.find(end_token)
                        # Only yield content before the end token
                        content_to_send = buffer[:end_pos]
                        if content_to_send:
                            yield f"data: {content_to_send}\n\n"
                        break  # Stop streaming
                    else:
                        # Safe to send content up to the potential start of end token
                        safe_length = len(buffer) - len(end_token) + 1
                        content_to_send = buffer[:safe_length]
                        buffer = buffer[safe_length:]  # Keep remaining part in buffer
                        if content_to_send:
                            accumulated_text += content_to_send
                            yield f"data: {content_to_send}\n\n"
                
                await asyncio.sleep(0.01)
            
            # If we've exited the loop without finding the end token,
            # send any remaining content in the buffer
            if buffer and end_token not in buffer:
                yield f"data: {buffer}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield "data: [Error] Streaming response failed.\n\n"
        
        logger.debug("Streaming completed")

    return StreamingResponse(event_generator(), media_type="text/event-stream")