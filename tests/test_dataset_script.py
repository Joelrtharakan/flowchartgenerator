from tests.dataset_script import (
    extract_prompt_mermaid,
    node_label_is_valid,
    parse_node_reference,
    normalize_mermaid,
    validate_dataset_item,
)


def test_node_label_is_valid():
    assert node_label_is_valid("Yes")
    assert node_label_is_valid("No")
    assert not node_label_is_valid("This label has way more than six words")


def test_parse_node_reference_variants():
    assert parse_node_reference("A([Start])") == ("A", "round", "Start")
    assert parse_node_reference("B[Step]") == ("B", "box", "Step")
    assert parse_node_reference("C{Is it ok?}") == ("C", "decision", "Is it ok?")


def test_normalize_mermaid_spacing_and_label_format():
    raw = "A[Start]-->B[Middle]\nB-->Z[End]"
    expected = "A[Start] --> B[Middle]\nB --> Z[End]"
    assert normalize_mermaid(raw) == expected


def test_extract_prompt_mermaid_from_messages():
    item = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Do the thing"},
            {"role": "assistant", "content": "flowchart TD\nA([Start]) --> Z([End])"},
        ]
    }
    prompt, mermaid = extract_prompt_mermaid(item)
    assert prompt == "Do the thing"
    assert "flowchart TD" in mermaid


def test_validate_dataset_item_valid_entry():
    item = {
        "prompt": "Draw a simple flowchart",
        "mermaid": "flowchart TD\nA([Start]) --> B[Step]\nB --> Z([End])",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Draw a simple flowchart"},
            {"role": "assistant", "content": "flowchart TD\nA([Start]) --> B[Step]\nB --> Z([End])"},
        ],
    }
    valid, reason = validate_dataset_item(item)
    assert valid, reason
