from enum import Enum
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class DeviceEnum(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

# Define a request schema using Pydantic
class TextGenerationPayload(BaseModel):
    """Payload for text generation parameters."""

    prompt: str = Field(
        ...,
        min_length=1,
        example="Hello, I'm an LLM",
        description="The text prompt to continue from."
    )
    max_length: int = Field(
        50,
        gt=0,
        le=1000,
        description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values produce more random outputs."
    )
    top_k: int = Field(
        50,
        ge=0,
        le=1000,
        description="Top-k filtering parameter."
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter."
    )
    do_sample: bool = Field(
        True,
        description="Enable sampling. If false, greedy search is used."
    )
    device: DeviceEnum = Field(
        DeviceEnum.cpu,
        description="Select between CPU, CUDA, or MPS for inference."
    )


# Load GPT-2 model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

app = FastAPI(
    title="GPT-2 Text Generation API",
    description="An API that generates text using the openai-community/gpt2 model.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(payload: TextGenerationPayload):
    """
    Generate text from a given prompt using GPT-2.
    """
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    

    # Validate device choice: check for CUDA and MPS availability
    if payload.device == DeviceEnum.cuda and not torch.cuda.is_available():
        raise HTTPException(
            status_code=400,
            detail="CUDA is not available but 'cuda' was requested."
        )
    if payload.device == DeviceEnum.mps and not torch.backends.mps.is_available():
        raise HTTPException(
            status_code=400,
            detail="MPS is not available but 'mps' was requested."
        )

    # Move the model to the chosen device
    device = payload.device.value
    model.to(device)
    input_ids = tokenizer.encode(payload.prompt, return_tensors="pt").to(device)

    if payload.max_length <= input_ids.shape[1]:
        raise HTTPException(status_code=400, detail="max_length must exceed prompt length.")

    output_ids = model.generate(
        input_ids,
        max_length=payload.max_length,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        do_sample=payload.do_sample
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)