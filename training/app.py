import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

with open("current_model.txt") as f:
    LORA_PATH = f.read().strip()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# LOAD MODEL (CPU)
# -----------------------
print("Loading model on CPU...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Model loaded!")

# -----------------------
# REQUEST FORMAT
# -----------------------
class Request(BaseModel):
    prompt: str

# -----------------------
# GENERATE FUNCTION
# -----------------------
def generate(prompt):
    text = f"<|user|>\nCreate a clean Mermaid flowchart with Start and End: {prompt}\n<|end|>\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "flowchart TD" in output:
        output = output.split("flowchart TD")[-1]
        output = "flowchart TD" + output

    return output.strip()

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/generate")
def generate_flowchart(req: Request):
    result = generate(req.prompt)
    return {"flowchart": result}