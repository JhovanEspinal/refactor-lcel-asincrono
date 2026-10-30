import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHAT_ANSWER = "Desorden molecular"


class FakeAPI:
    def __init__(self):
        self.requests: list[dict] = []
        self.http_failures_before_success: list[int] = []
        self.anthropic_text = CHAT_ANSWER
        self.openai_text = CHAT_ANSWER
        self.anthropic_stop_reason = "end_turn"
        self.openai_finish_reason = "stop"
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _make_handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body: bytes, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                if api.http_failures_before_success:
                    return self._fail(api.http_failures_before_success.pop(0))
                if self.path.endswith("/chat/completions"):
                    return self._openai()
                if self.path.endswith("/v1/messages"):
                    return self._anthropic(body)
                self._send(404, b"{}")

            def _fail(self, status):
                error = {"error": {"message": f"fallo {status}", "type": "x"}}
                if self.path.endswith("/v1/messages"):
                    error = {"type": "error", "error": {"type": "x", "message": f"fallo {status}"}}
                self._send(status, json.dumps(error).encode())

            def _openai(self):
                response = {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": api.openai_finish_reason,
                            "message": {"role": "assistant", "content": api.openai_text},
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
                self._send(200, json.dumps(response).encode())

            def _anthropic(self, body):
                response = {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": api.anthropic_text}],
                    "stop_reason": api.anthropic_stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                }
                self._send(200, json.dumps(response).encode())

        return Handler
