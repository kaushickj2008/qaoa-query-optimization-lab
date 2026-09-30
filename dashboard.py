"""Local-only research dashboard. Start: .venv312/bin/python dashboard.py.

Uses a bounded in-memory job registry and one worker. No authentication or
durable storage: do not expose this development server to a public network.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4

from dashboard_service import PROBLEMS, run_experiment, validate

ROOT = Path(__file__).resolve().parent / "web"
JOBS: dict[str, dict] = {}
LOCK = Lock()
WORKER = ThreadPoolExecutor(max_workers=1)


def execute(job_id: str, config: dict) -> None:
    try:
        result = run_experiment(config)
        # Convert NumPy arrays before making the result visible to HTTP threads.
        result = json.loads(json.dumps(result, default=lambda x: x.tolist(), allow_nan=False))
        with LOCK:
            JOBS[job_id] = {"status": "complete", "result": result}
    except Exception as error:
        with LOCK:
            JOBS[job_id] = {"status": "failed", "error": f"Experiment failed: {type(error).__name__}: {error}"}


class Handler(BaseHTTPRequestHandler):
    """Same-origin JSON API and explicit static-file allowlist; never serves repo files."""

    def send(self, status: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def reply(self, status: int, data: dict) -> None:
        self.send(status, json.dumps(data, allow_nan=False).encode())

    def local_request(self) -> bool:
        port = self.server.server_address[1]
        if self.headers.get("Host") not in {f"localhost:{port}", f"127.0.0.1:{port}"}:
            self.reply(403, {"error": "Local host required."})
            return False
        return True

    def do_GET(self) -> None:
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/problems":
            return self.reply(200, {"problems": PROBLEMS})
        if path.startswith("/api/jobs/"):
            with LOCK:
                job = JOBS.get(path.removeprefix("/api/jobs/"))
            return self.reply(200 if job else 404, job or {"error": "Run not found; the server may have restarted."})
        files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                 "/style.css": ("style.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
        if path not in files:
            return self.reply(404, {"error": "Not found."})
        name, kind = files[path]
        self.send(200, (ROOT / name).read_bytes(), kind)

    def do_POST(self) -> None:
        if not self.local_request():
            return
        if self.headers.get("Origin") != f"http://{self.headers['Host']}":
            return self.reply(403, {"error": "Same-origin request required."})
        if self.path != "/api/jobs":
            return self.reply(404, {"error": "Not found."})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "JSON required."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request body must be 1–4096 bytes.")
            config = validate(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeError) as error:
            return self.reply(400, {"error": str(error)})
        with LOCK:
            if any(job["status"] == "running" for job in JOBS.values()):
                return self.reply(409, {"error": "An experiment is already running. Please wait for it to finish."})
            if len(JOBS) >= 20:
                del JOBS[next(iter(JOBS))]
            job_id = uuid4().hex
            JOBS[job_id] = {"status": "running"}
            WORKER.submit(execute, job_id, config)
        self.reply(202, {"id": job_id, "status": "running"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"QAOA Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        WORKER.shutdown(wait=True)
