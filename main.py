from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define a request schema using Pydantic
class TextGenerationPayload(BaseModel):
    prompt: str = Field(..., description="The text prompt to continue from.")

# Load GPT-2 model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(payload: TextGenerationPayload):
    prompt = payload.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)