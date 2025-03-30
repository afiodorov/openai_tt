#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["python-dotenv", "websockets", "sounddevice", "asyncstdlib", "numpy"]
# ///
import argparse
import asyncio
import base64
import json
import logging
import os
from typing import Any

import websockets
from asyncstdlib import enumerate
from dotenv import load_dotenv

from audio import AudioStreamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def send_audio(ws, audio_stream: AudioStreamer, audio_done: asyncio.Event):
    async for i, chunk in enumerate(audio_stream):
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(message))

        if i % 20 == 0 and i > 0:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        if i % 40 == 0 and i > 0:
            await ws.send(
                json.dumps(
                    {"type": "response.create", "response": {"modalities": ["text"]}},
                ),
            )

    audio_done.set()


async def receive_transcripts(ws, audio_done: asyncio.Event):
    interim_text = ""
    while not audio_done.is_set():
        try:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
        except TimeoutError:
            continue

        if message.get("type") == "response.text.delta":
            # Append the new word (or token) with a space
            interim_text += message["delta"] + " "
            # Print the interim text using carriage return to overwrite the current line
            # logger.debug("\r" + interim_text)
        elif message.get("type") == "response.text.done":
            # Clear the current line using an ANSI escape sequence
            # (\033[K clears to end-of-line)
            logger.info("\r\033[K%s", message["text"])


async def transcribe_live_audio(
    *,
    uri,
    additional_headers: dict[str, str],
    config: dict[str, Any],
):
    async with websockets.connect(uri, additional_headers=additional_headers) as ws:
        _ = json.loads(await ws.recv())
        await ws.send(json.dumps(config))
        _ = json.loads(await ws.recv())
        async with AudioStreamer() as audio_stream, asyncio.TaskGroup() as tg:
            done = asyncio.Event()
            tg.create_task(receive_transcripts(ws, done))
            tg.create_task(send_audio(ws, audio_stream, done))


async def main():
    parser = argparse.ArgumentParser(
        description="Live audio transcription and translation",
    )
    parser.add_argument(
        "-s",
        "--source_lang",
        type=str,
        required=True,
        help="Source language",
    )
    parser.add_argument(
        "-t",
        "--target_lang",
        type=str,
        required=True,
        help="Target language",
    )
    args = parser.parse_args()

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

    additional_headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "OpenAI-Beta": "realtime=v1",
        "OpenAI-Log-Session": "1",
    }

    instructions = (
        f"You are a live translator. Translate from {args.source_lang} to "
        f"{args.target_lang} in real time. Do not provide any source text or "
        f"commentary—only the translation in {args.target_lang}. If there is no "
        f"speech, output '...'. Strive for coherent, natural translations that "
        f"connect smoothly with previous output."
    )

    config: dict[str, Any] = {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": instructions,
            "input_audio_format": "pcm16",
            "turn_detection": None,
            "input_audio_transcription": None,
            # "turn_detection": {
            #     "create_response": True,
            #     "type": "semantic_vad",
            #     "eagerness": "high",
            # },
            "tool_choice": "auto",
            "temperature": 0.8,
            "max_response_output_tokens": "inf",
        },
    }

    await transcribe_live_audio(
        uri=uri,
        additional_headers=additional_headers,
        config=config,
    )


if __name__ == "__main__":
    asyncio.run(main())
