import asyncio
import pyaudio
from threading import Thread
import signal
from typing import Optional

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24_000
CHUNK = 1024 * 10


class AudioStreamer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_streaming = False
        self.worker_thread: Optional[Thread] = None
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._old_sigint_handler = None
        self._closed = False  # internal flag to run close() only once

    def start_stream(self):
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.is_streaming = True
        print("Microphone stream started")
        self.worker_thread = Thread(target=self._reader, daemon=True)
        self.worker_thread.start()

    def _reader(self):
        # Background thread that continuously reads audio and pushes it to the asyncio queue.
        while self.is_streaming:
            if self.stream is None:
                break  # safeguard for type checking
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                if self.loop is not None:
                    # Push data into the asyncio queue.
                    asyncio.run_coroutine_threadsafe(self.queue.put(data), self.loop)
            except Exception as e:
                print("Error reading audio:", e)
                break

    def stop_stream(self):
        self.is_streaming = False
        if self.stream is not None:
            try:
                self.stream.stop_stream()
            except Exception as e:
                print("Error stopping stream:", e)
            self.stream.close()
            self.stream = None
        print("Microphone stream stopped")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop_stream()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
        self.p.terminate()
        print("Audio resources released")
        # Put a sentinel value (None) to ensure the async iterator exits.
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)

    def _handle_sigint(self, signum, frame):
        if self._closed:
            return
        print("KeyboardInterrupt received, shutting down stream...")
        self.close()

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        # Install our custom SIGINT handler.
        self._old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.start_stream()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Restore the previous SIGINT handler.
        if self._old_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._old_sigint_handler)
        self.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
