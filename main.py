import base64
import json
import os
import asyncio
import websockets
from typing import Any

from audio import AudioStreamer
from dotenv import load_dotenv


# Adjust this as needed: how many chunks to accumulate per message.
ACCUMULATE_COUNT = 100


async def send_audio(ws, audio_stream: AudioStreamer, audio_done: asyncio.Event):
    async for chunk in audio_stream:
        encoded_audio = base64.b64encode(chunk).decode()
        message = {
            "type": "input_audio_buffer.append",
            "audio": encoded_audio,
        }
        await ws.send(json.dumps(message))

    audio_done.set()


async def receive_transcripts(ws, audio_done: asyncio.Event):
    while not audio_done.is_set():
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=0.5)
            print("Transcript:", message)
        except asyncio.TimeoutError:
            # No message received within the timeout, loop and check the event.
            continue


async def transcribe_live_audio(
    *, uri, additional_headers: dict[str, str], config: dict[str, Any]
):
    async with websockets.connect(uri, additional_headers=additional_headers) as ws:
        session_data = json.loads(await ws.recv())
        print(session_data)
        await ws.send(json.dumps(config))
        print(json.loads(await ws.recv()))
        async with AudioStreamer() as audio_stream:
            async with asyncio.TaskGroup() as tg:
                done = asyncio.Event()
                tg.create_task(send_audio(ws, audio_stream, done))
                tg.create_task(receive_transcripts(ws, done))


async def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    uri = "wss://api.openai.com/v1/realtime?intent=transcription"

    additional_headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
        "OpenAI-Log-Session": "1",
    }
    config: dict[str, Any] = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe-latest",
                "prompt": "",
                "language": "en",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "input_audio_noise_reduction": {"type": "near_field"},
            "include": ["item.input_audio_transcription.logprobs"],
        },
    }

    await transcribe_live_audio(
        uri=uri, additional_headers=additional_headers, config=config
    )


if __name__ == "__main__":
    asyncio.run(main())
