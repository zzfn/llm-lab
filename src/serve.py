"""OpenAI 兼容服务接口 + 本地零依赖聊天/续写网页。

支持接口：
  GET  /                      本地聊天网页（对话 & 续写双模式）
  GET  /v1/models             返回模型列表
  GET  /v1/info               返回当前存档阶段、步数与参数量
  POST /v1/chat/completions   对话补全（支持 stream: true SSE 推送与 OpenAI tool_calls）
  POST /v1/completions        基座续写
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .generate import stream_ids
from .tokenizer import ByteBPETokenizer
from .utils import format_chat_prompt, load_ckpt, parse_tool_calls, pick_device

PAGE = Path(__file__).parent / "web" / "chat.html"
BASE_STAGES = ("pretrain", "anneal")


def to_internal_messages(messages: list[dict], tools: list | None = None) -> list[dict]:
    """将 OpenAI Chat 格式（含 tools 与 tool_calls）转换为内部格式。"""
    out: list[dict] = []
    if tools:
        specs = [t.get("function", t) for t in tools]
        out.append({
            "role": "system",
            "content": "你可以调用下面的工具，调用时输出 JSON：\n" + json.dumps(specs, ensure_ascii=False),
        })
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        msg: dict = {"role": role, "content": content}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", c)
                args = fn.get("arguments") or {}
                calls.append({"name": fn.get("name"), "arguments": json.loads(args) if isinstance(args, str) else args})
            msg["tool_calls"] = calls
        if out and out[0]["role"] == "system" and role == "system" and len(out) == 1 and tools:
            out[0]["content"] = msg["content"] + "\n" + out[0]["content"]
            continue
        out.append(msg)
    return out


class Service:
    def __init__(self, model, tok: ByteBPETokenizer, info: dict):
        self.model = model
        self.tok = tok
        self.info = info
        self.lock = threading.Lock()
        self.device = next(model.parameters()).device

    def _run_tokens(self, prompt_ids: list[int], max_tokens: int, temperature: float, top_k: int, stop_id: int) -> list[int]:
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out: list[int] = []
        for tid in stream_ids(self.model, x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, eos_id=stop_id):
            out.append(tid)
        return out

    def chat(self, body: dict) -> dict:
        msgs = to_internal_messages(body.get("messages", []), body.get("tools"))
        prompt_ids = format_chat_prompt(self.tok, msgs)
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 0.0))
        top_k = int(body.get("top_k", 40))

        with self.lock:
            gen = self._run_tokens(prompt_ids, max_tokens, temperature, top_k, self.tok.im_end_id)

        finished = bool(gen and gen[-1] == self.tok.im_end_id)
        text, calls, _ = parse_tool_calls(self.tok, gen)
        message: dict = {"role": "assistant", "content": text}
        finish_reason = "stop" if finished else "length"
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                }
                for c in calls
            ]
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.info["name"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(gen), "total_tokens": len(prompt_ids) + len(gen)},
        }

    def chat_stream(self, body: dict, write_sse) -> None:
        msgs = to_internal_messages(body.get("messages", []), body.get("tools"))
        prompt_ids = format_chat_prompt(self.tok, msgs)
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 0.0))
        top_k = int(body.get("top_k", 40))
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        write_sse({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.info["name"],
                   "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        with self.lock:
            x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            buf: list[int] = []
            prev_text = ""
            finished = False
            for tid in stream_ids(self.model, x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, eos_id=self.tok.im_end_id):
                if tid == self.tok.im_end_id:
                    finished = True
                    break
                buf.append(tid)
                cur_text = self.tok.decode(buf, skip_special=False)
                delta = cur_text[len(prev_text) :]
                if delta and "\ufffd" not in delta:
                    prev_text = cur_text
                    if not write_sse({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.info["name"],
                                      "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}):
                        return
        write_sse({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.info["name"],
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if finished else "length"}]})

    def complete(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        prompt_ids = [self.tok.bos_id] + self.tok.encode_ordinary(prompt)
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 0.8))
        top_k = int(body.get("top_k", 40))
        with self.lock:
            gen = self._run_tokens(prompt_ids, max_tokens, temperature, top_k, self.tok.eos_id)
        finished = bool(gen and gen[-1] == self.tok.eos_id)
        text = self.tok.decode(gen, skip_special=True)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.info["name"],
            "choices": [{"index": 0, "text": text, "finish_reason": "stop" if finished else "length"}],
        }


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _json(self, code: int, data: dict):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": svc.info["name"], "object": "model"}]})
            elif self.path == "/v1/info":
                self._json(200, svc.info)
            else:
                self._json(404, {"error": f"not found: {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/v1/chat/completions":
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    def write_sse(obj: dict) -> bool:
                        try:
                            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            return True
                        except OSError:
                            return False

                    svc.chat_stream(body, write_sse)
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
                else:
                    self._json(200, svc.chat(body))
            elif self.path == "/v1/completions":
                self._json(200, svc.complete(body))
            else:
                self._json(404, {"error": f"not found: {self.path}"})

    return Handler


def main():
    p = argparse.ArgumentParser(description="启动 OpenAI 兼容服务与离线聊天网页")
    p.add_argument("ckpt", nargs="?", default="out/rl/ckpt.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    tok = ByteBPETokenizer.load(args.tokenizer)
    model, ckpt = load_ckpt(args.ckpt, device)
    stage = ckpt.get("stage", "unknown")
    info = {
        "name": "llm-lab",
        "ckpt": args.ckpt,
        "stage": stage,
        "step": ckpt.get("step", 0),
        "params": model.count_parameters(),
        "max_seq_len": model.cfg.max_seq_len,
        "default_mode": "complete" if stage in BASE_STAGES else "chat",
    }
    svc = Service(model, tok, info)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(svc))
    print(f"服务已启动: http://{args.host}:{args.port}/  (stage={stage}, params={info['params']/1e6:.2f}M)")
    server.serve_forever()


if __name__ == "__main__":
    main()
