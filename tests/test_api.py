from unittest.mock import MagicMock
from app.services.stt_engine import STTResult
from app.services.youtube_downloader import DownloadResult
from app.services.summarizer import SummaryResult
from pathlib import Path

def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_summarize_endpoint(client, mock_pipeline, mock_summarizer):
    # Setup mocks
    mock_pipeline.run.return_value = (
        DownloadResult(
            url="http://test.url",
            title="Test Video",
            video_id="123",
            audio_path=Path("test.mp3")
        ),
        STTResult(
            text="This is a transcript.",
            segments=[],
            language="en"
        )
    )
    
    mock_summarizer.summarize_transcript.return_value = SummaryResult(
        final_summary="Final summary.",
        chunk_summaries=["Chunk 1"]
    )

    # Make request
    response = client.post("/api/summarize", json={"url": "http://test.url"})

    # Verify
    assert response.status_code == 200
    data = response.json()
    assert data["final_summary"] == "Final summary."
    assert data["title"] == "Test Video"
    
    # Verify mocks called
    mock_pipeline.run.assert_called_once()
    mock_summarizer.summarize_transcript.assert_called_once()
