from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "This is the starting point."}

@app.post("/predict")
def predict():
    return {"generated_text": "This is a dummy response for now."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
