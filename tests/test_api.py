import os
import sys

import pytest

os.environ["SKIP_MODEL_LOAD"] = "1"
os.environ["LOG_DIR"] = os.path.join(os.path.dirname(__file__), "_logs")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from fastapi.testclient import TestClient  # noqa: E402

import app as api  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_root_reports_running(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["status"]


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": False, "adapter": None}


def test_generate_returns_503_when_model_not_loaded(client):
    r = client.post("/generate", json={"prompt": "login flow"})
    assert r.status_code == 503


def test_generate_validates_body(client):
    r = client.post("/generate", json={})
    assert r.status_code == 422


def test_generate_with_stub_model(client, monkeypatch):
    monkeypatch.setattr(api, "model", object())
    monkeypatch.setattr(api, "generate", lambda p: "flowchart TD\nA([Start]) --> B([End])")
    r = client.post("/generate", json={"prompt": "login flow"})
    assert r.status_code == 200
    assert r.json()["flowchart"].startswith("flowchart TD")


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_logs_endpoints(client):
    assert client.get("/logs/api").status_code == 200
    assert client.get("/logs/monitor").status_code == 200


def test_read_logs_missing_file():
    assert api.read_logs("/nonexistent/file.log") == "No logs found"


def test_extract_mermaid_plain_finetuned_output():
    assert api.extract_mermaid("flowchart TD\nA([Start]) --> B([End])") == "flowchart TD\nA([Start]) --> B([End])"


def test_extract_mermaid_base_model_fenced_graph():
    raw = "```mermaid\ngraph TD\n    A[Start] --> B[End]\n```\nMore chatter about constraints"
    assert api.extract_mermaid(raw) == "flowchart TD\n    A[Start] --> B[End]"


def test_extract_mermaid_unterminated_fence():
    assert api.extract_mermaid("```mermaid\ngraph LR\nA --> B").startswith("flowchart LR")


def test_extract_mermaid_drops_text_before_flowchart():
    assert api.extract_mermaid("Sure! flowchart TD\nA --> B") == "flowchart TD\nA --> B"


def test_single_exit_decision_becomes_box():
    code = "flowchart TD\nA([Start]) --> B{Select Tenant}\nB -->|Choose Tenant| C[Tenant Workflow]\nC --> D([End])"
    fixed = api.fix_single_exit_decisions(code)
    assert "B[Select Tenant]" in fixed
    assert "{" not in fixed


def test_real_decision_is_kept():
    code = "flowchart TD\nA([Start]) --> B{Valid?}\nB -->|Yes| C[Go]\nB -->|No| D[Stop]\nC --> E([End])\nD --> E"
    assert api.fix_single_exit_decisions(code) == code


def test_inline_decision_definition_counts_exits():
    code = "flowchart TD\nA([Start]) --> B{OK?}\nB -->|Yes| C[Done]\nB -->|No| A"
    assert "B{OK?}" in api.fix_single_exit_decisions(code)


def test_extract_mermaid_strips_dangling_closing_fence():
    assert api.extract_mermaid("flowchart TD\nA([Start]) --> Z([End])\n```") == "flowchart TD\nA([Start]) --> Z([End])"
