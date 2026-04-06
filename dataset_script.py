# generate_dataset.py

import argparse
import json
import logging
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import requests
from requests import RequestException
from tqdm import tqdm

DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434/api/generate",
    "model": "qwen2.5-coder:3b",
    "output_dir": "data",
    "target": 500,
    "checkpoint_interval": 200,
    "retry_max_attempts": 3,
    "retry_base_delay": 1.0,
    "retry_max_delay": 30.0,
    "request_timeout": 90,
    "temperature": 0.75,
    "max_tokens": 900,
    "delay_between_calls": 0.05,
    "train_ratio": 0.90,
    "val_ratio": 0.05,
    "test_ratio": 0.05,
}

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

LOG_FILE_NAME = "dataset.log"
CHECKPOINT_FILE_NAME = "checkpoint_raw.jsonl"
ARROW_PATTERN = re.compile(
    r"^(?P<src>[A-Za-z0-9_]+)(?:\s*\[[^\]]*\]|\s*\([^\)]*\)|\s*\{[^\}]*\}|\s*/[^/]*/)?\s*-->\s*(?:\|[^|]*\|\s*)?(?P<dst>[A-Za-z0-9_]+)(?:\s*\[[^\]]*\]|\s*\([^\)]*\)|\s*\{[^\}]*\}|\s*/[^/]*/)?$"
)
NODE_ID_PATTERN = re.compile(r"(?P<id>[A-Za-z0-9_]+)(?=\s*(?:\[|\(|\{|/|$))")

logger = logging.getLogger(__name__)
_shutdown_requested = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Mermaid flowchart dataset from Ollama responses."
    )
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"], help="Directory to write logs, checkpoints, and output files.")
    parser.add_argument("--target", type=int, default=DEFAULT_CONFIG["target"], help="Number of valid flowchart pairs to generate.")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CONFIG["checkpoint_interval"], help="Save checkpoint every N valid results.")
    parser.add_argument("--ollama-url", default=DEFAULT_CONFIG["ollama_url"], help="Base Ollama API URL.")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"], help="Ollama model to use.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"], help="Sampling temperature for generation.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CONFIG["max_tokens"], help="Maximum number of tokens to request from the model.")
    parser.add_argument("--retry-max-attempts", type=int, default=DEFAULT_CONFIG["retry_max_attempts"], help="Maximum number of Ollama retry attempts.")
    parser.add_argument("--delay-between-calls", type=float, default=DEFAULT_CONFIG["delay_between_calls"], help="Seconds to wait between API calls.")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_CONFIG["request_timeout"], help="HTTP request timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint if one exists.")
    parser.add_argument(
        "--prompt-reuse",
        action="store_true",
        help="Allow reusing prompts after the prompt pool is exhausted.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(output_dir / LOG_FILE_NAME),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
    )


def call_ollama(
    prompt: str,
    url: str,
    model: str,
    request_timeout: int,
    temperature: float,
    max_tokens: int,
    retry_max_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> str:
    full_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
    last_error = ""

    for attempt in range(1, retry_max_attempts + 1):
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": ["User:", "System:"],
                    },
                },
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("response", "")
            if not isinstance(text, str):
                raise ValueError("Ollama response does not contain a string 'response' field")
            return text.strip()
        except RequestException as exc:
            last_error = f"request exception: {exc}"
            logger.warning("Attempt %d failed: %s", attempt, exc)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON response: {exc}"
            logger.warning("Attempt %d failed: invalid JSON", attempt)
        except ValueError as exc:
            last_error = f"invalid response payload: {exc}"
            logger.warning("Attempt %d failed: invalid payload", attempt)

        if attempt < retry_max_attempts:
            delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
            logger.info("Retrying in %.1fs...", delay)
            time.sleep(delay)

    logger.error("All %d attempts failed: %s", retry_max_attempts, last_error)
    return ""


def clean_mermaid(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            segment = part.strip()
            if segment.startswith("mermaid"):
                return segment[len("mermaid") :].strip()
            if "flowchart" in segment:
                return segment.strip()
    return raw.strip()


def build_prompt_pool() -> List[str]:
    return [template.format(s=scenario) for scenario in SCENARIOS for template in TEMPLATES]


def prompt_generator(prompt_pool: List[str], allow_reuse: bool) -> Iterator[str]:
    if not prompt_pool:
        raise ValueError("Prompt pool is empty")
    shuffled = prompt_pool.copy()
    random.shuffle(shuffled)
    while True:
        for prompt in shuffled:
            yield prompt
        if not allow_reuse:
            break
        random.shuffle(shuffled)


def build_messages(prompt: str, mermaid: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You generate Mermaid flowcharts. Return ONLY valid Mermaid code. No explanation."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": mermaid},
    ]


def normalize_entry(item: Dict[str, str]) -> Dict[str, str]:
    if "messages" not in item or item.get("messages") is None:
        prompt = item.get("prompt", "")
        mermaid = item.get("mermaid", "")
        item["messages"] = build_messages(prompt, mermaid)
    return item


def load_checkpoint(output_dir: Path) -> List[Dict[str, str]]:
    checkpoint_path = output_dir / CHECKPOINT_FILE_NAME
    if not checkpoint_path.exists():
        return []

    loaded: List[Dict[str, str]] = []
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                loaded.append(normalize_entry(item))

    if loaded:
        logger.info("Resuming from checkpoint with %d existing pairs", len(loaded))
    return loaded


def save_checkpoint(data: List[Dict[str, str]], output_dir: Path) -> None:
    checkpoint_path = output_dir / CHECKPOINT_FILE_NAME
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(normalize_entry(item), ensure_ascii=False) + "\n")


def extract_node_ids(line: str) -> Set[str]:
    return {match.group("id") for match in NODE_ID_PATTERN.finditer(line)}


def validate(mermaid: str) -> Tuple[bool, str]:
    mermaid = mermaid.strip()
    if "flowchart TD" not in mermaid:
        return False, "missing 'flowchart TD' declaration"
    if "A([Start])" not in mermaid:
        return False, "missing A([Start]) node"
    if "Z([End])" not in mermaid:
        return False, "missing Z([End]) node"
    if "-->" not in mermaid:
        return False, "no arrows/connections found"

    lines = [line.strip() for line in mermaid.splitlines() if line.strip()]
    if len(lines) < 3:
        return False, f"too few lines ({len(lines)})"
    if not lines[0].startswith("flowchart TD"):
        return False, "first nonempty line must start with 'flowchart TD'"

    graph: Dict[str, Set[str]] = {}
    defined_nodes: Set[str] = set()

    for line in lines[1:]:
        if line.count("[") != line.count("]"):
            return False, "unbalanced brackets"
        if line.count("{") != line.count("}"):
            return False, "unbalanced braces"
        if len(line) > 200:
            return False, "line exceeds 200 characters"

        if "-->" in line:
            match = ARROW_PATTERN.match(line)
            if not match:
                return False, f"invalid arrow syntax: {line}"
            src, dst = match.group("src"), match.group("dst")
            graph.setdefault(src, set()).add(dst)
            defined_nodes.update({src, dst})
        else:
            defined_nodes.update(extract_node_ids(line))

    if "A" not in defined_nodes or "Z" not in defined_nodes:
        return False, "required node IDs A and Z are missing"

    reachable = set()
    queue = ["A"]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        for neighbor in graph.get(current, set()):
            if neighbor not in reachable:
                queue.append(neighbor)

    if "Z" not in reachable:
        return False, "Z([End]) is not reachable from A([Start])"

    orphan_nodes = defined_nodes - reachable
    if orphan_nodes - {"A", "Z"}:
        return False, f"orphan nodes not reachable: {sorted(orphan_nodes)}"

    return True, ""


def save_splits(results: List[Dict[str, str]], output_dir: Path) -> None:
    random.shuffle(results)
    n = len(results)
    splits = {
        "train": results[: int(n * DEFAULT_CONFIG["train_ratio"])],
        "val": results[int(n * DEFAULT_CONFIG["train_ratio"]) : int(n * (DEFAULT_CONFIG["train_ratio"] + DEFAULT_CONFIG["val_ratio"]))],
        "test": results[int(n * (DEFAULT_CONFIG["train_ratio"] + DEFAULT_CONFIG["val_ratio"])) :],
    }
    for split_name, data in splits.items():
        path = output_dir / f"data_{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in data:
                messages = item.get("messages")
                if messages is None:
                    messages = build_messages(item.get("prompt", ""), item.get("mermaid", ""))
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(data)} pairs  →  {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    prompt_pool = build_prompt_pool()
    allow_reuse = args.prompt_reuse or args.target > len(prompt_pool)
    if args.target > len(prompt_pool) and not args.prompt_reuse:
        logger.warning(
            "Target %d exceeds distinct prompt pool size %d. Prompts will be reused once the pool is exhausted.",
            args.target,
            len(prompt_pool),
        )
        allow_reuse = True

    if args.resume:
        results = load_checkpoint(output_dir)
    else:
        results = []

    seen_mermaid = {item["mermaid"] for item in results}
    seen_prompts: Set[str] = set()
    generator = prompt_generator(prompt_pool, allow_reuse)

    def _signal_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("Shutdown requested - saving checkpoint before exit...")
        save_checkpoint(results, output_dir)
        logger.info("Checkpoint saved. Exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    pbar = tqdm(total=args.target, desc="Generating flowcharts")
    pbar.update(len(results))
    attempts = 0

    while len(results) < args.target and not _shutdown_requested:
        prompt = next(generator)
        if prompt in seen_prompts:
            continue

        attempts += 1
        raw = call_ollama(
            prompt=prompt,
            url=args.ollama_url,
            model=args.model,
            request_timeout=args.request_timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retry_max_attempts=args.retry_max_attempts,
            retry_base_delay=DEFAULT_CONFIG["retry_base_delay"],
            retry_max_delay=DEFAULT_CONFIG["retry_max_delay"],
        )
        if not raw:
            continue

        mermaid = clean_mermaid(raw)
        if mermaid in seen_mermaid:
            logger.debug("Duplicate Mermaid output skipped")
            continue

        is_valid, reason = validate(mermaid)
        if is_valid:
            messages = [
                {"role": "system", "content": "You generate Mermaid flowcharts. Return ONLY valid Mermaid code. No explanation."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": mermaid},
            ]
            results.append({"prompt": prompt, "mermaid": mermaid, "messages": messages})
            seen_prompts.add(prompt)
            seen_mermaid.add(mermaid)
            pbar.update(1)
        else:
            logger.debug("Invalid output from prompt '%s': %s", prompt, reason)

        if len(results) > 0 and len(results) % args.checkpoint_interval == 0:
            save_checkpoint(results, output_dir)
            logger.info("Checkpoint saved at %d pairs", len(results))

        time.sleep(args.delay_between_calls)

    pbar.close()
    pass_rate = 0.0 if attempts == 0 else 100 * len(results) / attempts
    print(f"\nGenerated {len(results)} valid pairs from {attempts} attempts ({pass_rate:.1f}% pass rate)")

    save_checkpoint(results, output_dir)
    save_splits(results, output_dir)


if __name__ == "__main__":
    main()
