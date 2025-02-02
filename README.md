# GPT-2 Text Generation API

A lightweight API to generate text using GPT-2, built with FastAPI and Hugging Face's Transformers.

## Features

- **Text Generation:** Generate continuations from a given prompt.
- **Configurable Options:** Adjust max length, temperature, top-k, and top-p sampling.
- **Flexible Device Support:** Choose between CPU, CUDA, or MPS.

## Requirements

- Python 3.8+
   ```bash
   requirements.txt

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/pierrehogenhaug/PFA.git
   cd PFA
2. **Install dependencies:**   
   ```bash
   pip install -r requirements.txt

## Running the API
1. Locally
Simply run the main.py script:
   ```bash
   python main.py
This starts the FastAPI server on http://localhost:8000.

2. In a Virtual Environment
- Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
- Install dependencies:
   ```bash
   pip install -r requirements.txt
- Launch the app:
   ```bash
   python main.py

3. Using Docker
- Build the Docker image:
   ``bash
docker build -t pfacase .
- Run the container:
   ```bash
   docker run -p 8000:8000 pfacase
Access the API at http://localhost:8000.

## How it works
`/ (Root)`
A simple endpoint that returns a greeting to confirm the service is running.
`/health`

A quick health check endpoint that returns:
{ "status": "healthy" }

`/predict`
Send a POST request with a JSON payload to generate text. Example payload:
{
  "prompt": "Hello, I'm an LLM",
  "max_length": 50,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 1.0,
  "do_sample": true,
  "device": "cpu"
}
The response will include the generated text:

{ "generated_text": "..." }

## Testing

To run tests for the API:
pytest
