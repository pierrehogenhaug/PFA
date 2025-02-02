# GPT-2 Text Generation API

A lightweight API to generate text using GPT-2, built with FastAPI and Hugging Face's Transformers.

## Features

- **Text Generation:** Generate continuations from a given prompt.
- **Configurable Options:** Adjust max length, temperature, top-k, and top-p sampling.
- **Flexible Device Support:** Choose between CPU, CUDA, or MPS.

## Requirements

- Python 3.8+
- [fastapi==0.95.0](https://pypi.org/project/fastapi/)
- [uvicorn==0.21.1](https://pypi.org/project/uvicorn/)
- [transformers==4.26.1](https://pypi.org/project/transformers/)
- [torch==2.0.1](https://pypi.org/project/torch/)

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. **Install dependencies:**   
   ```bash
   pip install -r requirements.txt
3. Running the API
