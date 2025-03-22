import asyncio
import sounddevice as sd
import numpy as np
import signal
import queue
from threading import Thread
from typing import Optional

CHANNELS = 1
SAMPLE_RATE = 24_000
CHUNK = 24_000 // 5


class AudioStreamer:
    def __init__(self):
        self.async_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._blocking_queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.worker_thread: Optional[Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._old_sigint_handler = None
        self._closed = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print("Stream status:", status)
        # Convert the NumPy array (indata) to bytes
        self._blocking_queue.put(indata.tobytes())

    def start_stream(self):
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=CHUNK,
            callback=self._audio_callback,
        )
        self.stream.start()
        print("SoundDevice InputStream started")
        # Start a background thread to transfer data from blocking queue to asyncio queue.
        self.worker_thread = Thread(target=self._queue_worker, daemon=True)
        self.worker_thread.start()

    def _queue_worker(self):
        # Continuously transfer audio data from the blocking queue to the asyncio queue.
        while not self._closed:
            try:
                data = self._blocking_queue.get(timeout=0.1)
                if self.loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        self.async_queue.put(data), self.loop
                    )
            except queue.Empty:
                continue

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._closed = True
        print("SoundDevice InputStream stopped")

    def close(self):
        self.stop_stream()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
        print("Audio resources released")
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.async_queue.put(None), self.loop)

    def _handle_sigint(self, signum, frame):
        print("KeyboardInterrupt received, shutting down stream...")
        self.close()

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        self._old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.start_stream()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._old_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._old_sigint_handler)
        self.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.async_queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
