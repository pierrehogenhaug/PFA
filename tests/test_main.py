import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_valid_request():
    """
    Test that a valid request to the /predict endpoint returns a 200 status
    and includes the 'generated_text' field in the response.
    """
    payload = {
        "prompt": "Hello, I'm an LLM",
        "max_length": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "device": "cpu",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "generated_text" in json_response
    assert isinstance(json_response["generated_text"], str)


def test_predict_empty_prompt():
    """
    Test that providing an empty prompt (or only whitespace) results in a 400 error.
    """
    payload = {
        "prompt": "   ",
        "max_length": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "device": "cpu",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Prompt must not be empty."


def test_predict_invalid_device():
    """
    Test that specifying a device that isn't loaded returns a 400 error.
    (For example, 'cuda' if CUDA is not available on the system.)
    """
    # We can change 'cuda' -> 'mps' if you'd like to test that scenario, 
    # assuming it's unavailable on your system
    payload = {
        "prompt": "Hello, I'm an LLM",
        "max_length": 50,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "device": "cuda",
    }

    response = client.post("/predict", json=payload)
    # On systems without CUDA, this should fail. If CUDA is available, 
    # we could switch 'device': 'mps' (if Apple Silicon isn't available), etc.
    assert response.status_code == 400
    assert "is not available" in response.json()["detail"]


def test_predict_max_length_too_small():
    """
    Test that if the max_length doesn't exceed the prompt length,
    the endpoint returns a 400 error.
    """
    short_max_length = 2
    payload = {
        "prompt": "Testing short max length",
        "max_length": short_max_length,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
        "device": "cpu",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "must exceed prompt length" in response.json()["detail"]