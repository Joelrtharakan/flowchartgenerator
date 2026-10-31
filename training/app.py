import os
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")  # pin a commit hash in production
LOG_DIR = os.getenv("LOG_DIR", ".")
CURRENT_MODEL_FILE = os.getenv("CURRENT_MODEL_FILE", "current_model.txt")
# float32 needs ~16 GB RAM for Phi-3 mini; use bfloat16 on smaller machines
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float32")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "250"))
# Set SKIP_MODEL_LOAD=1 to start the API without the model (CI tests, smoke checks)
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

API_LOG = os.path.join(LOG_DIR, "api.log")
MONITOR_LOG = os.path.join(LOG_DIR, "monitor.log")

# -----------------------
# LOGGING SETUP
# -----------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=API_LOG,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Ensure log files exist
open(API_LOG, "a").close()
open(MONITOR_LOG, "a").close()

# -----------------------
# FASTAPI INIT
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics (request count, latency, in-flight requests)
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# -----------------------
# LOAD MODEL (CPU)
# -----------------------
tokenizer = None
model = None
ADAPTER_NAME = None


def load_model():
    global tokenizer, model, ADAPTER_NAME

    # Heavy imports live here so the API can be imported without torch installed
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Read current model path (LoRA adapter); without one, serve the base model
    lora_path = None
    if os.path.exists(CURRENT_MODEL_FILE):
        with open(CURRENT_MODEL_FILE) as f:
            lora_path = f.read().strip()
        if not os.path.isdir(lora_path):
            logging.warning("LoRA adapter %s not found, using base model", lora_path)
            lora_path = None
    else:
        logging.warning("%s not found, using base model", CURRENT_MODEL_FILE)

    print("Loading model on CPU...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)  # nosec B615 - revision set via MODEL_REVISION

    base_model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - revision set via MODEL_REVISION
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=getattr(torch, MODEL_DTYPE),
        device_map="cpu"
    )

    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path)
        ADAPTER_NAME = lora_path
        print(f"Loaded LoRA adapter: {lora_path}")
    else:
        model = base_model
        print("No LoRA adapter found - serving the BASE model (not fine-tuned)")
    model.eval()  # IMPORTANT

    print("Model loaded!")


if not SKIP_MODEL_LOAD:
    load_model()

# -----------------------
# REQUEST FORMAT
# -----------------------
class Request(BaseModel):
    prompt: str

# -----------------------
# OUTPUT CLEANUP
# -----------------------
def extract_mermaid(output: str) -> str:
    """Return clean Mermaid source from raw model output.

    Handles the fine-tuned model (plain "flowchart TD ...") and the base model
    (markdown fence, "graph TD", trailing chatter).
    """
    fenced = re.search(r"```(?:mermaid)?[ \t]*\n(.*?)(?:```|$)", output, re.S)
    if fenced:
        output = fenced.group(1)

    # Drop a dangling closing fence (left by the stop string) and anything after it
    output = re.sub(r"\n?[ \t]*```.*$", "", output, flags=re.S)

    output = re.sub(r"^\s*graph\s+(TD|TB|LR|RL|BT)\b", r"flowchart \1", output)

    # Clean output to ensure valid Mermaid
    if "flowchart TD" in output:
        output = "flowchart TD" + output.split("flowchart TD")[-1]

    return fix_single_exit_decisions(output.strip())


_ARROW = re.compile(r"\s*-->(?:\|[^|]*\|)?\s*")


def fix_single_exit_decisions(code: str) -> str:
    """Turn decision nodes {..} with fewer than two outgoing edges into plain boxes.

    A diamond with a single exit is not a decision; it only confuses readers.
    """
    exits = {}
    for line in code.splitlines():
        parts = _ARROW.split(line)
        for part in parts[:-1]:
            m = re.match(r"\s*([A-Za-z0-9_]+)", part)
            if m:
                exits[m.group(1)] = exits.get(m.group(1), 0) + 1

    def repair(m):
        node_id, label = m.group(1), m.group(2)
        return m.group(0) if exits.get(node_id, 0) >= 2 else f"{node_id}[{label}]"

    return re.sub(r"\b([A-Za-z0-9_]+)\{([^{}\n]*)\}", repair, code)


# -----------------------
# GENERATE FUNCTION
# -----------------------
# Guidance for the base model (the fine-tuned adapter was trained on the short prompt instead)
BASE_PROMPT = """Create a Mermaid flowchart for the process below.
Rules:
- Start with: flowchart TD
- First node is A([Start]); last node is Z([End]).
- Use [Step] boxes for actions and {Question?} diamonds ONLY for real decisions.
- Every diamond needs two or more exits, labelled like -->|Yes| and -->|No|.
- 6 to 12 nodes, labels of at most 5 words, every branch must reach End.
- Output only the Mermaid code.

Example process: Order checkout
```mermaid
flowchart TD
A([Start]) --> B[Add items to cart]
B --> C[Enter payment details]
C --> D{Payment valid?}
D -->|Yes| E[Confirm order]
D -->|No| F[Show error]
F --> C
E --> Z([End])
```

Process: """


def generate(prompt: str):
    import torch

    if ADAPTER_NAME:
        text = f"<|user|>\nCreate a clean Mermaid flowchart with Start and End: {prompt}\n<|end|>\n<|assistant|>\n"
        prefill, stop = "", None
    else:
        # Prefill the opening of the code fence so the model continues straight into the diagram
        text = f"<|user|>\n{BASE_PROMPT}{prompt}\n<|end|>\n<|assistant|>\n```mermaid\nflowchart TD\n"
        prefill, stop = "flowchart TD\n", ["```"]

    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            **({"stop_strings": stop, "tokenizer": tokenizer} if stop else {})
        )

    # Decode only the newly generated tokens (not the echoed prompt)
    output = prefill + tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return extract_mermaid(output)

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/generate")
def generate_flowchart(req: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

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
    except OSError:
        return "No logs found"

@app.get("/logs/api", response_class=PlainTextResponse)
def get_api_logs():
    return read_logs(API_LOG)

@app.get("/logs/monitor", response_class=PlainTextResponse)
def get_monitor_logs():
    return read_logs(MONITOR_LOG)

# -----------------------
# ROOT CHECK
# -----------------------
@app.get("/")
def home():
    return {"status": "API is running 🚀"}

# -----------------------
# HEALTH CHECK (liveness / readiness probes)
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "adapter": ADAPTER_NAME}
