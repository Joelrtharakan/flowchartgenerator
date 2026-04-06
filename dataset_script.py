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
    "model": "qwen2.5-coder:7b",
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
8. Do NOT use D[/.../] or E[/.../] node syntax
9. Do NOT add text labels such as "End" to arrows
10. Keep all labels under 6 words
11. Use consistent arrow spacing: C -->|Yes| D
12. Every path must reach Z([End]) — no dead ends
13. Do not use extra trailing spaces on any line
14. Keep node style consistent across the entire flowchart"""

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
    parser.add_argument("--validate-dataset", action="store_true", help="Validate JSONL dataset files in the output directory.")
    parser.add_argument("--clean-dataset", action="store_true", help="Clean JSONL dataset files in the output directory by removing invalid examples.")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Overwrite dataset files when cleaning them. Otherwise writes cleaned copies with .cleaned.jsonl suffix.")
    parser.add_argument("--dataset-prefix", default="data_", help="Prefix for dataset JSONL files to validate or clean.")
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


def normalize_mermaid(mermaid: str) -> str:
    lines = []
    for raw_line in mermaid.splitlines():
        line = raw_line.rstrip()
        if "-->" in line:
            line = re.sub(r"\s*-->\s*", " --> ", line)
            line = re.sub(r"-->\s*\|", "-->|", line)
            line = re.sub(r"\|\s*(.*?)\s*\|", r"|\1|", line)
            line = re.sub(r" {2,}", " ", line)
        else:
            line = line.strip()
        lines.append(line)
    return "\n".join(lines).strip()


def clean_mermaid(raw: str) -> str:
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            segment = part.strip()
            if segment.startswith("mermaid"):
                return normalize_mermaid(segment[len("mermaid") :].strip())
            if "flowchart" in segment:
                return normalize_mermaid(segment.strip())
    return normalize_mermaid(raw)


def enforce_inline_format(code: str) -> str:
    import re
    lines = code.split("\n")
    
    node_defs = {}
    node_pattern = re.compile(r"([A-Za-z0-9_]+)(\[.*?\]|\(\[.*?\]\)|\(.*?\)|\{.*?\})")
    
    for line in lines:
        for match in node_pattern.finditer(line):
            node_defs[match.group(1)] = match.group(2)
            
    filtered = []
    used_nodes = set()
    
    for line in lines:
        if line.startswith("flowchart"):
            continue
            
        if "-->" in line:
            def replacer(match):
                node_id = match.group(1)
                
                if node_id in node_defs and node_id not in used_nodes:
                    used_nodes.add(node_id)
                    return f"{node_id}{node_defs[node_id]}"
                return node_id
            
            new_line = re.sub(r"([A-Za-z0-9_]+)(\[.*?\]|\(\[.*?\]\)|\(.*?\)|\{.*?\})?", replacer, line)
            filtered.append(new_line)

    if lines and lines[0].startswith("flowchart"):
        return lines[0].strip() + "\n" + "\n".join(filtered)

    return "\n".join(filtered)


def repair_mermaid(raw: str) -> str:
    mermaid = clean_mermaid(raw)
    lines = []

    for raw_line in mermaid.splitlines():
        line = raw_line.strip()
        if "-->" not in line:
            lines.append(line)
            continue

        # normalize whitespace and arrow formatting first
        line = normalize_mermaid(line)

        # repair invalid node syntax [/text/] to [text]
        line = re.sub(r"\[/([^/]+?)/\]", r"[\1]", line)
        line = re.sub(r"/([^/]+?)/", r"[\1]", line)

        # normalize valid Yes/No labels to exact spacing
        line = re.sub(r"\s*-->\s*\|\s*(Yes|No)\s*\|", r" -->|\1|", line)

        # remove invalid arrow labels that are not Yes/No
        def replace_label(match: re.Match) -> str:
            label = match.group(1).strip()
            if label in {"Yes", "No"}:
                return f"|{label}|"
            return ""

        line = re.sub(r"\s*\|\s*(.*?)\s*\|", replace_label, line)
        line = re.sub(r" -->\s*\|", " -->|", line)

        # if label removal left consecutive arrows, clean them back up
        line = re.sub(r"-->\|\|", "-->", line)
        line = re.sub(r"\|\|", "", line)
        line = re.sub(r"(?<!\|)\s*-->\s*(?!\|)", " --> ", line)
        line = re.sub(r" {2,}", " ", line).strip()

        lines.append(line)

    mermaid = "\n".join(lines).strip()

    # enforce inline formatting
    mermaid = enforce_inline_format(mermaid)

    return mermaid


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
            target_item = normalize_entry(item)
            handle.write(json.dumps({"messages": target_item["messages"]}, ensure_ascii=False) + "\n")


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
        if "/" in line:
            return False, "node syntax /.../ is not allowed"
        if "-->" not in line:
            return False, "standalone node definitions not allowed"

        match = ARROW_PATTERN.match(line)
        if not match:
            return False, f"invalid arrow syntax: {line}"
        src, dst = match.group("src"), match.group("dst")
        graph.setdefault(src, set()).add(dst)
        defined_nodes.update({src, dst})

        if "|" in line:
            labels = re.findall(r"\|(.*?)\|", line)
            if not labels:
                return False, f"invalid arrow label syntax: {line}"
            for label in labels:
                if label not in {"Yes", "No"}:
                    return False, f"invalid arrow label '{label}'"

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
                prompt = item.get("prompt") or find_prompt_in_messages(item.get("messages", []))
                mermaid = item.get("mermaid") or find_mermaid_in_messages(item.get("messages", []))
                mermaid = repair_mermaid(mermaid)
                messages = build_messages(prompt, mermaid)
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(data)} pairs  →  {path}")


def find_prompt_in_messages(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return ""
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def find_mermaid_in_messages(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def is_valid_message_list(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in {"system", "user", "assistant"}:
            return False
        if not isinstance(msg.get("content"), str):
            return False
    return True


def validate_dataset_item(item: Dict[str, object]) -> Tuple[bool, str]:
    prompt = item.get("prompt") if isinstance(item.get("prompt"), str) else find_prompt_in_messages(item.get("messages", []))
    mermaid = item.get("mermaid") if isinstance(item.get("mermaid"), str) else find_mermaid_in_messages(item.get("messages", []))
    if not prompt:
        return False, "missing prompt"
    if not mermaid:
        return False, "missing Mermaid content"

    cleaned = clean_mermaid(mermaid)
    valid, reason = validate(cleaned)
    if not valid:
        return False, reason

    if "messages" in item:
        if not is_valid_message_list(item["messages"]):
            return False, "invalid messages format"
        if len(item["messages"]) < 3:
            return False, "messages list must contain system, user, and assistant"
        if item["messages"][0].get("role") != "system":
            return False, "first message role must be system"
        if item["messages"][1].get("role") != "user":
            return False, "second message role must be user"
        if item["messages"][-1].get("role") != "assistant":
            return False, "last message role must be assistant"
    return True, ""


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def validate_dataset_file(path: Path) -> None:
    entries = load_jsonl(path)
    invalid_count = 0
    for index, entry in enumerate(entries, start=1):
        entry = normalize_entry(entry)
        valid, reason = validate_dataset_item(entry)
        if not valid:
            invalid_count += 1
            logger.info("Invalid entry %d in %s: %s", index, path.name, reason)
    if invalid_count == 0:
        print(f"{path.name}: all {len(entries)} entries are valid")
    else:
        print(f"{path.name}: {invalid_count} invalid entries found out of {len(entries)}")


def clean_dataset_file(path: Path, overwrite: bool = False) -> None:
    entries = load_jsonl(path)
    cleaned: List[Dict[str, object]] = []
    invalid_count = 0
    for entry in entries:
        item = normalize_entry(entry)
        prompt = item.get("prompt") if isinstance(item.get("prompt"), str) else find_prompt_in_messages(item.get("messages", []))
        mermaid = item.get("mermaid") if isinstance(item.get("mermaid"), str) else find_mermaid_in_messages(item.get("messages", []))
        mermaid = repair_mermaid(mermaid)
        item["prompt"] = prompt
        item["mermaid"] = mermaid
        item["messages"] = build_messages(prompt, mermaid)
        valid, reason = validate_dataset_item(item)
        if valid:
            cleaned.append(item)
        else:
            invalid_count += 1
            logger.info("Dropping invalid entry in %s: %s", path.name, reason)

    target_path = path if overwrite else path.with_suffix(path.suffix + ".cleaned.jsonl")
    with target_path.open("w", encoding="utf-8") as handle:
        for item in cleaned:
            handle.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")

    print(f"{path.name}: cleaned {len(cleaned)} valid entries, removed {invalid_count} invalid entries, output -> {target_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    if args.validate_dataset or args.clean_dataset:
        dataset_files = sorted(output_dir.glob(f"{args.dataset_prefix}*.jsonl"))
        if not dataset_files:
            logger.warning("No dataset files found in %s", output_dir)
            return

        for dataset_path in dataset_files:
            if args.clean_dataset:
                clean_dataset_file(dataset_path, overwrite=args.overwrite_dataset)
            else:
                validate_dataset_file(dataset_path)
        return

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

    seen_mermaid = {item.get("mermaid") or find_mermaid_in_messages(item.get("messages", [])) for item in results}
    seen_prompts = {item.get("prompt") or find_prompt_in_messages(item.get("messages", [])) for item in results}
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

        mermaid = repair_mermaid(raw)
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
