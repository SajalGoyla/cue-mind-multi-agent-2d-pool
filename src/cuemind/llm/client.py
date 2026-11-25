"""LLM clients behind one small interface.

* OllamaClient: a local open-weight model served by Ollama (default qwen2.5:7b-instruct).
* FakeLLM: deterministic offline stand-in used by every test and by the demo's offline mode.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from cuemind.llm.cache import ResponseCache, request_key


@dataclass
class LLMResult:
    parsed: BaseModel | None
    raw_text: str
    stop_reason: str | None
    usage: dict[str, float] = field(default_factory=dict)  # input_tokens, output_tokens
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


# ----------------------------------------------------------------------------- ollama
class OllamaClient:
    """Local open-weight model served by Ollama (http://127.0.0.1:11434).

    Uses /api/chat with `format` = the pydantic JSON schema (constrained decoding), a fixed
    sampling temperature and a seed derived from repeat_idx, so runs are reproducible.
    Responses are cached on disk (see cache.py); offline=True answers from the cache only.
    """

    def __init__(self, model: str | None = None, *, host: str | None = None,
                 temperature: float = 0.7, num_ctx: int = 8192, timeout_s: float = 300.0,
                 cache: ResponseCache | None = None, offline: bool = False):
        self.model = model or os.environ.get("CUEMIND_OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.effort = f"T{temperature}"
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.temperature, self.num_ctx, self.timeout_s = temperature, num_ctx, timeout_s
        self.cache, self.offline = cache, offline

    RETRY_DELAYS_S = (5, 10, 20, 30, 30, 30, 30, 30)  # ~3 min: survives a server restart

    def _post_with_retry(self, httpx, body: dict) -> dict:
        """Infrastructure failures (server down/restarting, 5xx, timeouts) are retried with
        backoff instead of surfacing as a model failure (D-027)."""
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
            except httpx.TransportError as exc:  # connect/read errors, resets, timeouts
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

        key = request_key(model=self.model, effort=self.effort, system=system, user=user,
                          schema=schema.__name__, repeat_idx=repeat_idx)
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
        except Exception as exc:  # noqa: BLE001 - invalid JSON / schema mismatch / timeout
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = LLMResult(parsed, raw, stop, usage, time.perf_counter() - t0, False, err,
                           self.model, key)
        if self.cache is not None and err is None:  # never cache failures
            self.cache.put(key, result.log())
        return result


# ----------------------------------------------------------------------------- fake
_ROW = re.compile(r"^(S\d{2}) \| (.*)$", re.M)
_EASE = {"easy geometry": 3, "moderate geometry": 2, "hard geometry": 1}
_ROTATE = {"offense": 0, "position": 1, "defense": 2}


def _row_score(text: str) -> int:
    return next((v for k, v in _EASE.items() if k in text), 0)


class FakeLLM:
    """Deterministic offline LLM. Reads the candidate lines (`Sxx | ...`) from the user message
    and ranks them by the difficulty word (easy > moderate > hard > safeties), then by id.

    mode="best":     every agent follows that ranking (specialists agree)
    mode="disagree": each specialist rotates the ranking, so their first choices differ
    mode="invalid":  returns an unknown shot id (exercises retry + fallback)
    mode="error":    returns no parsed output
    """

    def __init__(self, mode: str = "best", model: str = "fake-llm"):
        self.mode, self.model = mode, model
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult:
        self.calls.append({"agent": agent, "schema": schema.__name__, "repeat_idx": repeat_idx})
        usage = {"input_tokens": len(system + user) // 4, "output_tokens": 50}
        if self.mode == "error":
            return LLMResult(None, "", "stop", usage, 0.0, error="fake error", model=self.model)
        ranked = [sid for sid, text in sorted(_ROW.findall(user), key=lambda r: (-_row_score(r[1]), r[0]))]
        if not ranked:
            return LLMResult(None, "", "stop", usage, 0.0, error="no candidates", model=self.model)
        if self.mode == "invalid":
            ranked = ["S99"]
        name = schema.__name__
        if name == "SpecialistShortlist":
            if self.mode == "disagree":
                k = _ROTATE.get(agent, 0) % len(ranked)
                ranked = ranked[k:] + ranked[:k]
            obj = {"shortlist": [{"shot_id": s, "reason": f"fake {agent} view"} for s in ranked[:3]]}
        elif name == "CoordinatorDecision":
            obj = {"shot_id": ranked[0], "intent": "offense", "rationale": "fake coordinator",
                   "confidence": 0.6, "followed": []}
        else:
            obj = {"shot_id": ranked[0], "intent": "offense", "rationale": "fake single agent",
                   "confidence": 0.6}
        return LLMResult(schema.model_validate(obj), "", "stop", usage, 0.0, model=self.model)
