# Offline YouTube Video Summarizer

An end-to-end AI system that downloads YouTube audio, transcribes it, and generates a concise summary—all running efficiently on your local machine without external APIs.

## Features

- **Privacy-First & Offline**: Uses local models for all processing. No data leaves your machine.
- **YouTube Downloader**: Extracts audio from YouTube videos using `yt-dlp`.
- **Offline Transcription**: Powered by `faster-whisper` for fast and accurate speech-to-text.
- **Local Summarization**: Uses `microsoft/Phi-4-mini-instruct` (via Hugging Face Transformers) for high-quality, abstractive summaries.
- **Concurrency control**: Prevents server overload by managing heavy model execution.

## System Architecture

1.  **Downloader**: `yt-dlp` fetches audio as an MP3/WAV file.
2.  **Transcription**: `faster-whisper` (an optimized implementation of OpenAI's Whisper) converts audio to text.
3.  **Summarization**:
    *   **Chunking**: Long transcripts are split into manageable chunks.
    *   **Map**: Each chunk is summarized individually by Phi-4.
    *   **Reduce**: Chunk summaries are combined into a final coherent summary.
4.  **API Layer**: FastAPI handles requests, manages model lifecycle, and ensures thread safety.

## Setup and Installation

### Prerequisites
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to your system PATH (required for audio processing).
- (Optional) NVIDIA GPU with CUDA for faster inference.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd yt-transcript-summarizer
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the API Server

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`.

- **Swagger UI**: Visit `http://localhost:8000/docs` to interact with the API.
- **Health Check**: `GET /api/health`

### 2. Summarize a Video (API)

**Endpoint**: `POST /api/summarize`

**Body**:
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

**Response**:
```json
{
  "final_summary": "The video discusses...",
  "transcript_length_chars": 1234,
  "detected_language": "en"
}
```

### 3. CLI Usage

You can also run the pipeline directly via the command line:

```bash
python -m scripts.summarize_youtube --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Options**:
- `--device`: `cuda` or `cpu`
- `--stt-model-size`: `tiny`, `small`, `base`, `medium`, `large-v3` (default: `small`)

## Design Choices & Justification

### STT: faster-whisper

**Reasoning:**
- Purely offline, open-source.
- Optimized with CTranslate2 for fast inference.
- Supports multiple model sizes to trade off speed vs. accuracy.

**Trade-offs:**
- Larger models (e.g., medium, large) give better accuracy but require more VRAM and time.
- *Decision*: For this project, `small` is a good balance for a standard 6 GB - 8 GB GPU.

### Summarization: microsoft/Phi-4-mini-instruct

**Reasoning:**
- Modern, instruction-tuned LLM that runs locally with `transformers`.
- Supports long context; works well for instruction-style summarization prompts.

**Trade-offs:**
- Heavier than classic summarization models like BART/T5 but far more powerful and flexible.
- *Decision*: To avoid OOM on limited VRAM, transcript is token-chunked and summarized hierarchically.

### Long-Transcript Handling

- The transcript is tokenized with the Phi-4 tokenizer and split into chunks of up to `max_input_tokens` (configurable, default 8000).
- Each chunk is summarized individually.
- Chunk summaries are combined and summarized again into a final global summary.
- This pattern respects context limits and stabilizes memory usage.

## Error Handling & Robustness

The application handles:
1.  **Invalid or unsupported URLs**: Returns `400 Bad Request` with a descriptive error.
2.  **Download failures**: Caught around `yt-dlp` (e.g., network issues, region restrictions) and surfaced as user-facing errors.
3.  **Very long videos**: Summarization pipeline is chunk-based; it will not exceed the configured token limit.
4.  **Model misconfigurations**: Errors from `faster-whisper` or `transformers` are wrapped and returned as `500` responses.

## Challenges Faced

1.  **Setting up offline STT and summarization on limited GPU memory**:
    *   *Solution*: Used `faster-whisper` (int8 quantization) and Phi-4 (4-bit or half-precision if applicable) to fit within consumer VRAM limits. Implemented `asyncio.Lock` to prevent concurrent heavy inference.

2.  **Balancing model size vs. latency vs. summary quality**:
    *   *Solution*: Chosen `small` Whisper model and `Phi-4-mini`. Large models were too slow for a synchronous-feeling API.

3.  **Handling long transcripts with a hierarchical summarization approach**:
    *   *Solution*: Implemented the Map-Reduce pattern. Splitting by *tokens* (not just characters) was crucial to avoid context window overflows.

4.  **Managing dependencies**:
    *   *Solution*: Created a `Dockerfile` that installs system-level dependencies (`ffmpeg`, `git`) before Python packages to ensure a smooth build process.

## Docker Usage

1.  **Build the image**:
    ```bash
    docker build -t yt-summarizer .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 8000:8000 yt-summarizer
    ```

Open `http://localhost:8000/docs` to test.
