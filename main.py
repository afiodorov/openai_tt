import argparse
import base64
import json
import os
import asyncio
import websockets
from typing import Any

from audio import AudioStreamer
from dotenv import load_dotenv


async def send_audio(ws, audio_stream: AudioStreamer, audio_done: asyncio.Event):
    async for chunk in audio_stream:
        message = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("utf-8"),
        }
        await ws.send(json.dumps(message))

    audio_done.set()


async def receive_transcripts(ws, audio_done: asyncio.Event):
    interim_text = ""
    while not audio_done.is_set():
        try:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            if message.get("type") == "response.text.delta":
                # Append the new word (or token) with a space
                interim_text += message["delta"] + " "
                # Print the interim text using carriage return to overwrite the current line
                print("\r" + interim_text, end="", flush=True)
            elif message.get("type") == "response.text.done":
                # Clear the current line using an ANSI escape sequence (\033[K clears to end-of-line)
                print("\r\033[K", end="", flush=True)
                # Print the final transcript on a new line
                print(message["text"])
        except asyncio.TimeoutError:
            # No message received within the timeout; continue checking.
            continue


async def transcribe_live_audio(
    *, uri, additional_headers: dict[str, str], config: dict[str, Any]
):
    async with websockets.connect(uri, additional_headers=additional_headers) as ws:
        _ = json.loads(await ws.recv())
        await ws.send(json.dumps(config))
        _ = json.loads(await ws.recv())
        async with AudioStreamer() as audio_stream, asyncio.TaskGroup() as tg:
            done = asyncio.Event()
            tg.create_task(send_audio(ws, audio_stream, done))
            tg.create_task(receive_transcripts(ws, done))


async def main():
    parser = argparse.ArgumentParser(
        description="Live audio transcription and translation"
    )
    parser.add_argument(
        "-s", "--source_lang", type=str, required=True, help="Source language"
    )
    parser.add_argument(
        "-t", "--target_lang", type=str, required=True, help="Target language"
    )
    args = parser.parse_args()

    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

    additional_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
        "OpenAI-Log-Session": "1",
    }
    config: dict[str, Any] = {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": f"Translate into {args.target_lang} from {args.source_lang}",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-transcribe"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True,
            },
            "tool_choice": "auto",
            "temperature": 0.8,
            "max_response_output_tokens": "inf",
        },
    }

    await transcribe_live_audio(
        uri=uri, additional_headers=additional_headers, config=config
    )


if __name__ == "__main__":
    asyncio.run(main())
