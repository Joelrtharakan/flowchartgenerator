import  torch
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# -----------------------
# LOGGING SETUP
# -----------------------
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Ensure log files exist
open("api.log", "a").close()
open("monitor.log", "a").close()

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

# Read current model path (LoRA adapter)
with open("current_model.txt") as f:
    LORA_PATH = f.read().strip()

# -----------------------
# FASTAPI INIT
# -----------------------
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
model.eval()  # IMPORTANT

print("Model loaded!")

# -----------------------
# REQUEST FORMAT
# -----------------------
class Request(BaseModel):
    prompt: str

# -----------------------
# GENERATE FUNCTION
# -----------------------
def generate(prompt: str):
    text = f"<|user|>\nCreate a clean Mermaid flowchart with Start and End: {prompt}\n<|end|>\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean output to ensure valid Mermaid
    if "flowchart TD" in output:
        output = output.split("flowchart TD")[-1]
        output = "flowchart TD" + output

    return output.strip()

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/generate")
def generate_flowchart(req: Request):
    logging.info(f"Prompt: {req.prompt}")

    result = generate(req.prompt)

    logging.info(f"Output: {result[:200]}")  # limit size

    return {"flowchart": result}

# -----------------------
# LOG READER
# -----------------------
def read_logs(file_path, lines=100):
    try:
        with open(file_path, "r") as f:
            return "".join(f.readlines()[-lines:])
    except:
        return "No logs found"

@app.get("/logs/api", response_class=PlainTextResponse)
def get_api_logs():
    return read_logs("api.log")

@app.get("/logs/monitor", response_class=PlainTextResponse)
def get_monitor_logs():
    return read_logs("monitor.log")

# -----------------------
# ROOT CHECK
# -----------------------
@app.get("/")
def home():
    return {"status": "API is running 🚀"}