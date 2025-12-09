import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_pipeline(mocker):
    # Mock the pipeline in app.state
    mock = MagicMock()
    app.state.pipeline = mock
    return mock

@pytest.fixture
def mock_summarizer(mocker):
    # Mock the summarizer in app.state
    mock = MagicMock()
    app.state.summarizer = mock
    return mock
