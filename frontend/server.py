import http.server
import socketserver
import urllib.request
import json
import os
import re

PORT = int(os.environ.get('PORT', 8082))
API_URL = os.environ.get('API_URL', "http://20.219.50.73:8000/generate")

def fix_mermaid(code: str) -> str:
    """
    Clean up truncated or malformed Mermaid output from the model:
    - Remove lines that are incomplete (unclosed brackets, dangling arrows).
    - Remove trailing whitespace/empty lines.
    """
    if not code:
        return code

    lines = code.strip().splitlines()
    clean = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Count opening vs closing brackets/parens to detect incomplete nodes
        opens  = sum(stripped.count(c) for c in ['[', '(', '{', '|'])
        closes = sum(stripped.count(c) for c in [']', ')', '}', '|'])
        # A line ending with a dangling arrow '-->` alone is also incomplete
        is_dangling_arrow = re.match(r'^[A-Za-z0-9_]+\s*-->', stripped) and stripped.endswith('-->')
        # Detect unclosed bracket: more opens than closes, and ends without a closing char
        is_unclosed = opens > closes and not stripped.endswith((']', ')', '}'))

        if is_dangling_arrow or is_unclosed:
            # Skip this incomplete line and stop processing further lines
            break

        clean.append(line)

    return '\n'.join(clean)


class ProxyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow local requests
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Proxy request to Azure
            req = urllib.request.Request(API_URL, data=post_data, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req) as response:
                    azure_response = response.read()

                # Parse, sanitize, and re-encode
                data = json.loads(azure_response)
                if 'flowchart' in data:
                    data['flowchart'] = fix_mermaid(data['flowchart'])
                clean_response = json.dumps(data).encode()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(clean_response)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Use socketserver with allow_reuse_address to avoid address bound errors on restart
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
    print(f"Proxy server running at http://127.0.0.1:{PORT}")
    print(f"Proxying /generate directly to {API_URL}")
    httpd.serve_forever()
