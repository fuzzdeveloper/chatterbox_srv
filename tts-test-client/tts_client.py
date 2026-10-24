#!/usr/bin/env python3
"""TTS Client — serves a local web UI and proxies TTS API calls to avoid CORS."""

import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urljoin, urlparse
import os
import sys

# ── Configuration ────────────────────────────────────────────────
UI_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
TTS_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:8880/")


class Handler(SimpleHTTPRequestHandler):
    """Serve the static UI page and proxy /api/* to the TTS server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    # ── routing ──────────────────────────────────────────────────
    def do_GET(self):
        if self._is_api():
            self._proxy_get()
        else:
            super().do_GET()

    def do_POST(self):
        if self._is_api():
            self._proxy_post()
        else:
            super().do_POST()

    # ── helpers ──────────────────────────────────────────────────
    def _api_prefix(self):
        return "/v1/audio"       # matches /v1/audio/voices, /v1/audio/speech

    def _is_api(self):
        return self.path.startswith(self._api_prefix())

    @property
    def upstream_url(self):
        return TTS_URL.rstrip("/") + "/"

    def _proxy_get(self):
        target = urljoin(self.upstream_url, self.path.lstrip("/"))
        req = urllib.request.Request(target, method="GET")
        self._forward(req)

    def _proxy_post(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        target = urljoin(self.upstream_url, self.path.lstrip("/"))
        req = urllib.request.Request(target, data=body, method="POST")
        # forward content-type
        ct = self.headers.get("Content-Type", "application/json")
        req.add_header("Content-Type", ct)
        self._forward(req)

    def _forward(self, req):
        """Send a request to the upstream TTS server and stream back the response."""
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            status = resp.status
            headers_raw = resp.headers
            data = resp.read()

            # Strip CORS-restricted headers that browsers ignore anyway
            self.send_response(status)
            for hdr in ("Content-Type", "Content-Length", "Transfer-Encoding"):
                val = headers_raw.get(hdr)
                if val:
                    self.send_header(hdr, val)
            self.end_headers()
            self.wfile.write(data)

        except urllib.error.HTTPError as e:
            err_data = e.read() if e.fp else b""
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.end_headers()
            self.wfile.write(err_data)

        except Exception as e:
            print(f"\n⚠ Proxy error: {e}\n", file=sys.stderr)
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Bad gateway: {e}".encode())


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"  TTS Server URL : {TTS_URL}")
    print(f"  Local UI port  : {UI_PORT}")
    print()

    server = ThreadedHTTPServer(("0.0.0.0", UI_PORT), Handler)
    try:
        print(f"✓  Open http://localhost:{UI_PORT}/ in your browser\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")


if __name__ == "__main__":
    main()
