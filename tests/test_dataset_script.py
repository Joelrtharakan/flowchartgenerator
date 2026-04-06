import json
from pathlib import Path

import pytest

from dataset_script import (
    SCENARIOS,
    TEMPLATES,
    build_prompt_pool,
    clean_dataset_file,
    clean_mermaid,
    prompt_generator,
    repair_mermaid,
    validate,
    validate_dataset_item,
)


def test_clean_mermaid_strips_markdown_fence():
    raw = "```mermaid\nflowchart TD\nA([Start]) --> B[Do thing]\n```"
    assert clean_mermaid(raw) == "flowchart TD\nA([Start]) --> B[Do thing]"


def test_clean_mermaid_returns_raw_when_no_fence():
    raw = "flowchart TD\nA([Start]) --> B[Do thing]"
    assert clean_mermaid(raw) == raw


def test_validate_accepts_simple_mermaid():
    mermaid = "flowchart TD\nA([Start]) --> B[Do thing]\nB --> Z([End])"
    valid, reason = validate(mermaid)
    assert valid, reason


def test_validate_rejects_missing_start():
    mermaid = "flowchart TD\nB[Do thing] --> Z([End])"
    valid, reason = validate(mermaid)
    assert not valid
    assert "A([Start])" in reason or "required node IDs" in reason


def test_validate_rejects_unreachable_end():
    mermaid = "flowchart TD\nA([Start]) --> B[Do thing]\nC[Other] --> Z([End])"
    valid, reason = validate(mermaid)
    assert not valid
    assert "not reachable" in reason


def test_clean_mermaid_normalizes_arrow_spacing():
    raw = "flowchart TD  \nA([Start])--> |Yes| B[Done]  "
    assert clean_mermaid(raw) == "flowchart TD\nA([Start]) -->|Yes| B[Done]"


def test_validate_rejects_io_node_syntax():
    mermaid = "flowchart TD\nA([Start]) --> B[/Upload/]\nB --> Z([End])"
    valid, reason = validate(mermaid)
    assert not valid
    assert "not allowed" in reason


def test_validate_rejects_end_label_on_arrow():
    mermaid = "flowchart TD\nA([Start]) --> B[Do thing]\nB -->|End| Z([End])"
    valid, reason = validate(mermaid)
    assert not valid
    assert "invalid arrow label" in reason


def test_prompt_generator_reuses_when_requested():
    pool = ["foo", "bar"]
    generator = prompt_generator(pool, allow_reuse=True)
    first_two = [next(generator), next(generator)]
    assert set(first_two) == {"foo", "bar"}
    third = next(generator)
    assert third in pool


def test_build_prompt_pool_contains_templates():
    pool = build_prompt_pool()
    assert len(pool) == len(SCENARIOS) * len(TEMPLATES)
    assert any("Create a flowchart for:" in prompt for prompt in pool)


def test_validate_dataset_item_rejects_bad_mermaid():
    item = {
        "messages": [
            {"role": "system", "content": "You generate Mermaid flowcharts."},
            {"role": "user", "content": "Create a flowchart for: product added to cart and checkout"},
            {"role": "assistant", "content": "flowchart TD  \nA([Start]) -->|Add Product| B[Product Added]  \nC-->|Yes| D[/Confirm Payment/]\nD -->|End| Z([End])"},
        ]
    }
    valid, reason = validate_dataset_item(item)
    assert not valid
    assert reason


def test_repair_mermaid_removes_invalid_nodes_and_labels():
    raw = "flowchart TD\nA([Start]) -->|Add Product| B[Product Added]\nB -->|Checkout Now| C{Is payment method selected?}\nC -->|Yes| D[/Confirm Payment/]\nD -->|End| Z([End])\nC -->|No| E[/Select Payment Method/]\nE -->|Retry Checkout| B"
    repaired = repair_mermaid(raw)
    assert "[/Confirm Payment/]" not in repaired
    assert "[/Select Payment Method/]" not in repaired
    assert "|Checkout Now|" not in repaired
    assert "|End|" not in repaired
    assert "C -->|No| E[Select Payment Method]" in repaired
    assert "C -->|Yes| D[Confirm Payment]" in repaired
    assert "D[Confirm Payment]" in repaired
    assert "E[Select Payment Method]" in repaired


def test_clean_dataset_file_filters_invalid_entries(tmp_path: Path):
    source = tmp_path / "data_test.jsonl"
    source.write_text(
        json.dumps({"messages": [
            {"role": "system", "content": "You generate Mermaid flowcharts."},
            {"role": "user", "content": "Create a flowchart for: product added to cart and checkout"},
            {"role": "assistant", "content": "flowchart TD\nA([Start]) --> B[Add Product]\nB --> Z([End])"},
        ]}
        ) + "\n" + json.dumps({"messages": [
            {"role": "system", "content": "You generate Mermaid flowcharts."},
            {"role": "user", "content": "Create a flowchart for: invalid sample"},
            {"role": "assistant", "content": "flowchart TD\nA([Start]) --> B[/Bad/]\nB --> Z([End])"},
        ]}
        ) + "\n",
        encoding="utf-8",
    )
    clean_dataset_file(source, overwrite=False)
    cleaned = tmp_path / "data_test.jsonl.cleaned.jsonl"
    assert cleaned.exists()
    lines = cleaned.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "B[Bad]" in lines[1]
