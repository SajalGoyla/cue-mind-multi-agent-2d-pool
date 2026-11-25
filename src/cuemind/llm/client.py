"""LLM clients behind one small interface (v2: local model only, D-029).

* OllamaClient: open-weight model served locally by Ollama (default qwen2.5:3b-instruct).
  JSON-schema constrained output, fixed temperature, seed per sample, response cache,
  retries on infrastructure failures (D-027).
* FakeLLM: deterministic offline stand-in used by the tests and the demo's offline mode.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from cuemind.llm.cache import ResponseCache, request_key

DEFAULT_MODEL = "qwen2.5:3b-instruct"


@dataclass
class LLMResult:
    parsed: BaseModel | None
    raw_text: str
    stop_reason: str | None
    usage: dict[str, float] = field(default_factory=dict)  # input/output tokens
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    model: str = ""
    request_hash: str = ""

    def log(self) -> dict[str, Any]:
        d = asdict(self)
        d["parsed"] = self.parsed.model_dump() if self.parsed is not None else None
        return d


class LLMClient(Protocol):
    model: str

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult: ...


class OllamaClient:
    RETRY_DELAYS_S = (5, 10, 20, 30, 30, 30, 30, 30)  # ~3 min: survives a server restart

    def __init__(self, model: str | None = None, *, host: str | None = None,
                 temperature: float = 0.7, num_ctx: int = 8192, timeout_s: float = 300.0,
                 cache: ResponseCache | None = None, offline: bool = False):
        self.model = model or os.environ.get("CUEMIND_OLLAMA_MODEL", DEFAULT_MODEL)
        self.temperature, self.num_ctx, self.timeout_s = temperature, num_ctx, timeout_s
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.cache, self.offline = cache, offline

    def _post_with_retry(self, httpx, body: dict) -> dict:
        """Infrastructure failures (server down/restarting, resets, 5xx, timeouts) are retried
        with backoff instead of surfacing as a model failure (D-027)."""
        last: Exception | None = None
        for delay in (0, *self.RETRY_DELAYS_S):
            if delay:
                time.sleep(delay)
            try:
                r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except httpx.TransportError as exc:
                last = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last = exc
        assert last is not None
        raise last

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult:
        import httpx

        key = request_key(model=self.model, temperature=self.temperature, system=system,
                          user=user, schema=schema.__name__, repeat_idx=repeat_idx)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            parsed = schema.model_validate(hit["parsed"]) if hit.get("parsed") else None
            return LLMResult(parsed=parsed, raw_text=hit.get("raw_text", ""),
                             stop_reason=hit.get("stop_reason"), usage=hit.get("usage", {}),
                             latency_s=hit.get("latency_s", 0.0), cached=True,
                             error=hit.get("error"), model=self.model, request_hash=key)
        if self.offline:
            return LLMResult(None, "", None, error="offline mode: response not in cache",
                             model=self.model, request_hash=key)
        body = {"model": self.model, "stream": False, "format": schema.model_json_schema(),
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "options": {"temperature": self.temperature, "seed": 1000 + repeat_idx,
                            "num_ctx": self.num_ctx}}
        t0 = time.perf_counter()
        parsed, raw, stop, usage, err = None, "", None, {}, None
        try:
            data = self._post_with_retry(httpx, body)
            raw = data.get("message", {}).get("content", "")
            stop = data.get("done_reason")
            usage = {"input_tokens": data.get("prompt_eval_count", 0) or 0,
                     "output_tokens": data.get("eval_count", 0) or 0}
            if stop == "length":
                err = "stop_reason=length (output truncated)"
            else:
                parsed = schema.model_validate_json(raw)
        except httpx.TransportError as exc:
            err = f"connection: is `ollama serve` running at {self.host}? ({exc})"
        except httpx.HTTPStatusError as exc:
            err = f"http_{exc.response.status_code}: {exc.response.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - invalid JSON / schema mismatch
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = LLMResult(parsed, raw, stop, usage, time.perf_counter() - t0, False, err,
                           self.model, key)
        if self.cache is not None and err is None:
            self.cache.put(key, result.log())
        return result


# ----------------------------------------------------------------------------- fake
_ROW = re.compile(r"^(S\d{2}) \| (.*)$", re.M)
_EASE = {"easy": 3, "moderate": 2, "hard": 1}


class FakeLLM:
    """Deterministic offline LLM. Reads the candidate table from the user message.

    mode="best":    picks the pot with the best ease word (ties: medium stun, lowest id)
    mode="first":   always the first row
    mode="invalid": returns an unknown shot id (exercises retry + fallback)
    mode="error":   returns no parsed output
    """

    def __init__(self, mode: str = "best", model: str = "fake-llm"):
        self.mode, self.model = mode, model
        self.calls: list[dict] = []

    @staticmethod
    def _rank(text: str) -> tuple:
        ease = next((v for k, v in _EASE.items() if f"{k} geometry" in text), 0)
        pot = text.startswith("Pot ")
        return (pot, ease, "medium speed, stun" in text)

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult:
        self.calls.append({"agent": agent, "schema": schema.__name__, "repeat_idx": repeat_idx})
        usage = {"input_tokens": len(system + user) // 4, "output_tokens": 40}
        if self.mode == "error":
            return LLMResult(None, "", "stop", usage, error="fake error", model=self.model)
        rows = _ROW.findall(user)
        if not rows:
            return LLMResult(None, "", "stop", usage, error="no candidates", model=self.model)
        ranked = sorted(rows, key=lambda r: (self._rank(r[1]), -int(r[0][1:])), reverse=True)
        sid = "S99" if self.mode == "invalid" else (rows[0][0] if self.mode == "first" else ranked[0][0])
        name = schema.__name__
        if name == "Shortlist":
            ids = [sid] + [r[0] for r in ranked if r[0] != sid][:2]
            obj = {"picks": [{"shot_id": i, "reason": f"fake {agent}"} for i in ids]}
        elif name == "CoordinatorDecision":
            obj = {"shot_id": sid, "rationale": "fake coordinator", "followed": ["offense"]}
        else:
            obj = {"shot_id": sid, "rationale": "fake single agent"}
        return LLMResult(schema.model_validate(obj), "", "stop", usage, 0.0, model=self.model)
