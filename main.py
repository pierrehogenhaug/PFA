from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define a request schema using Pydantic
class TextGenerationPayload(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
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
        description="Sampling temperature. Higher values lead to more randomness."
    )

# Load GPT-2 model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(payload: TextGenerationPayload):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    
    input_ids = tokenizer.encode(payload.prompt, return_tensors="pt")
    if payload.max_length <= input_ids.shape[1]:
        raise HTTPException(status_code=400, detail="max_length must exceed prompt length.")

    output_ids = model.generate(
        input_ids,
        max_length=payload.max_length,
        temperature=payload.temperature
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)