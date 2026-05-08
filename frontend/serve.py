"""Tiny static server for the v0.1 dashboard. Serves index.html on :3000.

    python frontend/serve.py
"""
from __future__ import annotations

import http.server
import os
import socketserver
import sys


def main(port: int = 3000) -> None:
    web_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_root)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Bracket dashboard at http://127.0.0.1:{port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    main(port)
