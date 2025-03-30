# OpenAI Streaming Transcription

A Python application using OpenAI's Realtime API to transcribe microphone input in real-time.

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- PortAudio (system dependency for sounddevice)

### Install PortAudio (system dependency)

On macOS:
```bash
brew install portaudio
```

On Linux:
```bash
sudo apt-get install portaudio19-dev
```

## Setup

1. Install dependencies using uv:
```bash
uv pip install -e .
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env to add your API key
```

## Usage

Run the transcription service:
```bash
uv run main.py -s english -t spanish
```

Press Ctrl+C to stop the streaming session.

## Development

1. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

2. Use the provided Makefile commands:
```bash
# Check code with ruff
make check

# Fix linting issues
make lint

# Format code
make format

# Run both linting and formatting
make lint-all
```