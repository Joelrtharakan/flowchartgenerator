import json
import os
import socketserver
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))

import server  # noqa: E402


def test_fix_mermaid_empty():
    assert server.fix_mermaid("") == ""


def test_fix_mermaid_keeps_complete_lines():
    code = "flowchart TD\nA([Start]) --> B[Step]\nB --> C([End])"
    assert server.fix_mermaid(code) == code


def test_fix_mermaid_drops_unclosed_node_and_after():
    code = "flowchart TD\nA([Start]) --> B[Step]\nB --> C[Unfinis"
    assert server.fix_mermaid(code) == "flowchart TD\nA([Start]) --> B[Step]"


def test_fix_mermaid_drops_dangling_arrow():
    code = "flowchart TD\nA([Start]) --> B[Step]\nB-->"
    assert server.fix_mermaid(code) == "flowchart TD\nA([Start]) --> B[Step]"


def _serve(handler):
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_health_and_unknown_post_route():
    httpd = _serve(server.ProxyHTTPRequestHandler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            assert json.loads(r.read()) == {"status": "ok"}

        req = urllib.request.Request(f"{base}/nope", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
