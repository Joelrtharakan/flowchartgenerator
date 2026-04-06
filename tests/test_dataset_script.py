from pathlib import Path

import pytest

from dataset_script import SCENARIOS, TEMPLATES, build_prompt_pool, clean_mermaid, prompt_generator, validate


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
