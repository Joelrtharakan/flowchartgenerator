# generate_dataset.py

import requests, json, random, time, os
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen2.5-coder:7b"
OUTPUT_DIR = "data"
TARGET     = 5000   # total pairs to generate — run overnight for more

os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are a Mermaid flowchart expert. When given a process description, you output ONLY valid Mermaid flowchart TD syntax — nothing else. No explanation, no markdown fences, no preamble.

Strict rules you always follow:
1. First line is always: flowchart TD
2. First node always: A([Start])
3. Last node always: Z([End])
4. Process steps use rectangles: B[Do something]
5. Decision points use diamonds: C{Is condition true?}
6. YES path labeled: -->|Yes|
7. NO path labeled: -->|No|
8. Input/Output steps: D[/Read input/]
9. Keep all labels under 6 words
10. Every path must reach Z([End]) — no dead ends
11. Use subgraph only if the prompt describes clear phases"""

SCENARIOS = [
    # Auth & accounts
    "user login with password and 2FA",
    "new user registration with email verification",
    "password reset via email link",
    "OAuth login with Google account",
    "session token refresh and expiry",
    "account deletion with 30-day grace period",
    "role-based access control check",
    "brute force login protection with lockout",
    # E-commerce
    "product added to cart and checkout",
    "credit card payment with fraud check",
    "order placed and sent to warehouse",
    "refund request and approval workflow",
    "coupon code validation at checkout",
    "inventory check before order confirmation",
    "subscription plan upgrade or downgrade",
    "digital product download after purchase",
    # DevOps & software
    "CI/CD pipeline from git push to deploy",
    "pull request review and merge process",
    "docker container build and push to registry",
    "automated test run with pass or fail gate",
    "database migration with rollback support",
    "API rate limiting per user per minute",
    "bug report triage and assignment",
    "release versioning and changelog generation",
    # Data & files
    "file upload with virus scan and storage",
    "image resize and thumbnail generation",
    "CSV import with validation and error rows",
    "database backup with verification step",
    "log rotation and archival process",
    "data pipeline from source to warehouse",
    # Customer support
    "support ticket creation and routing",
    "live chat escalation to human agent",
    "automated email response classification",
    "SLA breach detection and alert",
    "customer feedback survey after closure",
    # Healthcare & finance
    "patient appointment booking and reminder",
    "loan application with credit score check",
    "insurance claim submission and review",
    "ATM cash withdrawal with balance check",
    "payroll processing with tax deduction",
    # Everyday processes
    "food delivery order from placement to arrival",
    "hotel room booking with availability check",
    "taxi ride request and driver assignment",
    "flight check-in and boarding pass issue",
    "library book borrowing and return",
    "student exam submission and auto-grading",
    "employee leave request and manager approval",
    "recruitment application screening process",
    "inventory restocking when stock is low",
    "spam email detection and filtering",
]

TEMPLATES = [
    "Create a flowchart for: {s}",
    "Draw the complete process flow for {s}",
    "Map out all the steps and decisions for {s}",
    "Design a flowchart showing {s}",
    "Show the full decision flow when handling {s}",
]

def call_ollama(prompt: str) -> str:
    full = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model":   MODEL,
            "prompt":  full,
            "stream":  False,
            "options": {"temperature": 0.75, "num_predict": 900, "stop": ["User:", "System:"]}
        }, timeout=90)
        return resp.json().get("response", "").strip()
    except Exception as e:
        return ""

def clean_mermaid(raw: str) -> str:
    """Strip accidental markdown fences if the model adds them."""
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("mermaid"):
                return part[7:].strip()
            if "flowchart" in part:
                return part.strip()
    return raw.strip()

def validate(m: str) -> bool:
    """Basic structural validation."""
    if "flowchart TD" not in m:   return False
    if "Start"        not in m:   return False
    if "End"          not in m:   return False
    if "-->"          not in m:   return False
    lines = [l.strip() for l in m.splitlines() if l.strip()]
    if len(lines) < 4:            return False
    return True

results = []
attempts = 0
pbar = tqdm(total=TARGET, desc="Generating pairs")

while len(results) < TARGET:
    scenario = random.choice(SCENARIOS)
    prompt   = random.choice(TEMPLATES).format(s=scenario)
    attempts += 1

    raw     = call_ollama(prompt)
    mermaid = clean_mermaid(raw)

    if validate(mermaid):
        results.append({"prompt": prompt, "mermaid": mermaid})
        pbar.update(1)
    else:
        pass  # silently skip invalid outputs

    # Save checkpoint every 200 valid pairs
    if len(results) % 200 == 0 and len(results) > 0:
        with open(f"{OUTPUT_DIR}/checkpoint_raw.jsonl", "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

    time.sleep(0.05)

pbar.close()
print(f"\nGenerated {len(results)} valid pairs from {attempts} attempts ({100*len(results)/attempts:.1f}% pass rate)")

# Shuffle and split 90 / 5 / 5
random.shuffle(results)
n  = len(results)
splits = {
    "train": results[:int(n * 0.90)],
    "val":   results[int(n * 0.90):int(n * 0.95)],
    "test":  results[int(n * 0.95):]
}

for split, data in splits.items():
    path = f"{OUTPUT_DIR}/data_{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {split}: {len(data)} pairs  →  {path}")