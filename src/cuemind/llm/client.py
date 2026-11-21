"""LLM clients behind one small interface.

* ClaudeVertexClient: Claude through Google Cloud Vertex AI (anthropic SDK, AnthropicVertex),
  structured outputs via `messages.parse`, prompt caching on the system prompt, response
  cache, cross-process budget guard, token and cost accounting.
* FakeLLM: deterministic offline stand-in used by every test and by the demo's offline mode.

Notes on the API (verified against the installed SDK, anthropic 2.12):
* Haiku 5.5 / Sonnet 5.5 reject non-default temperature, so outputs are not bit-reproducible;
  the response cache gives exact replays, and repeat_idx lets us sample several answers.
* Thinking is adaptive by default; depth is controlled with output_config.effort.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel

from cuemind.llm.cache import Budget, BudgetExceeded, ResponseCache, request_key

DEFAULT_PRICES = {  # USD per million tokens; overridden by experiments/configs/prices.yaml
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10, "cache_write": 2.50},
}


def load_prices(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parents[3] / "experiments" / "configs" / "prices.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return {**DEFAULT_PRICES, **data.get("models", {})}
    return dict(DEFAULT_PRICES)


def cost_usd(model: str, usage: dict, prices: dict | None = None) -> float:
    pr = (prices or DEFAULT_PRICES).get(model)
    if pr is None:
        return 0.0
    return (usage.get("input_tokens", 0) * pr["input"] + usage.get("output_tokens", 0) * pr["output"]
            + usage.get("cache_read_tokens", 0) * pr["cache_read"]
            + usage.get("cache_write_tokens", 0) * pr["cache_write"]) / 1e6


@dataclass
class LLMResult:
    parsed: BaseModel | None
    raw_text: str
    stop_reason: str | None
    usage: dict[str, float] = field(default_factory=dict)  # tokens + cost_usd
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


def _usage_dict(u: Any) -> dict[str, float]:
    return {"input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}


class ClaudeVertexClient:
    def __init__(self, model: str | None = None, effort: str | None = None, *,
                 project_id: str | None = None, region: str | None = None,
                 max_tokens: int = 4000, timeout_s: float = 60.0,
                 cache: ResponseCache | None = None, budget: Budget | None = None,
                 prices: dict | None = None, offline: bool = False):
        self.model = model or os.environ.get("CUEMIND_MODEL", "claude-haiku-5-5")
        self.effort = effort or os.environ.get("CUEMIND_EFFORT", "medium")
        self.max_tokens, self.timeout_s = max_tokens, timeout_s
        self.cache, self.budget = cache, budget
        self.prices = prices or load_prices()
        self.offline = offline  # cache-only mode (demo without network)
        self._client = None
        self._project = project_id or os.environ.get("GCP_PROJECT_ID")
        self._region = region or os.environ.get("VERTEX_REGION", "global")

    def _sdk(self):
        if self._client is None:
            from anthropic import AnthropicVertex

            if not self._project:
                raise RuntimeError("GCP_PROJECT_ID is not set (see .env.example)")
            self._client = AnthropicVertex(project_id=self._project, region=self._region,
                                           max_retries=5, timeout=self.timeout_s)
        return self._client

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult:
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
        if self.budget is not None:
            self.budget.check()  # raises BudgetExceeded

        import anthropic

        t0 = time.perf_counter()
        parsed, raw, stop, usage, err = None, "", None, {}, None
        try:
            resp = self._sdk().messages.parse(
                model=self.model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema, output_config={"effort": self.effort},
            )
            stop = resp.stop_reason
            usage = _usage_dict(resp.usage)
            raw = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
            if stop in ("refusal", "max_tokens"):
                err = f"stop_reason={stop}"
            else:
                parsed = resp.parsed_output
                if parsed is None:
                    err = "no parsed output"
        except BudgetExceeded:
            raise
        except anthropic.RateLimitError as exc:
            err = f"rate_limited: {exc}"
        except anthropic.APIStatusError as exc:
            err = f"api_status_{exc.status_code}: {exc}"
        except anthropic.APIConnectionError as exc:
            err = f"connection: {exc}"
        except Exception as exc:  # noqa: BLE001 - schema validation etc.
            err = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - t0
        usage["cost_usd"] = cost_usd(self.model, usage, self.prices)
        if self.budget is not None and usage["cost_usd"]:
            self.budget.add(usage["cost_usd"])
        result = LLMResult(parsed, raw, stop, usage, latency, False, err, self.model, key)
        if self.cache is not None and err is None:  # never cache failures
            self.cache.put(key, result.log())
        return result


# ----------------------------------------------------------------------------- ollama
class OllamaClient:
    """Local open-weight model served by Ollama (http://127.0.0.1:11434).

    Uses /api/chat with `format` = the pydantic JSON schema (constrained decoding), a fixed
    sampling temperature and a seed derived from repeat_idx, so runs are reproducible and
    repeat_idx > 0 gives independent samples (SA-vote, E10). Cost is zero; tokens and
    latency are still recorded. Same cache and offline semantics as ClaudeVertexClient.
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
        backoff instead of surfacing as a model failure, which would otherwise be recorded as
        a fallback decision and bias the results (D-027)."""
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
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
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
                     "output_tokens": data.get("eval_count", 0) or 0,
                     "cache_read_tokens": 0, "cache_write_tokens": 0}
            if stop == "length":
                err = "stop_reason=length (output truncated)"
            else:
                parsed = schema.model_validate_json(raw)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            err = f"connection: is `ollama serve` running at {self.host}? ({exc})"
        except httpx.HTTPStatusError as exc:
            err = f"http_{exc.response.status_code}: {exc.response.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - invalid JSON / schema mismatch / timeout
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
        usage["cost_usd"] = 0.0
        result = LLMResult(parsed, raw, stop, usage, time.perf_counter() - t0, False, err,
                           self.model, key)
        if self.cache is not None and err is None:
            self.cache.put(key, result.log())
        return result


# ----------------------------------------------------------------------------- fake
_ROW = re.compile(r"^(S\d{2}) \| (.*)$", re.M)
_POT = re.compile(r"pots (\d+)%")


class FakeLLM:
    """Deterministic offline LLM. Reads the candidate table from the user message.

    mode="best":    picks the candidate with the highest 'pots NN%' (else the first row)
    mode="first":   always the first row
    mode="invalid": returns an unknown shot id (exercises retry + fallback)
    mode="error":   returns no parsed output
    """

    def __init__(self, mode: str = "best", model: str = "fake-llm"):
        self.mode, self.model = mode, model
        self.calls: list[dict] = []

    def _rows(self, user: str) -> list[tuple[str, float]]:
        rows = []
        for sid, text in _ROW.findall(user):
            m = _POT.search(text)
            rows.append((sid, float(m.group(1)) if m else 0.0))
        return rows

    def complete(self, *, system: str, user: str, schema: type[BaseModel], agent: str,
                 repeat_idx: int = 0) -> LLMResult:
        self.calls.append({"agent": agent, "schema": schema.__name__, "repeat_idx": repeat_idx})
        usage = {"input_tokens": len(system + user) // 4, "output_tokens": 50,
                 "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0}
        if self.mode == "error":
            return LLMResult(None, "", "end_turn", usage, 0.0, error="fake error", model=self.model)
        rows = self._rows(user)
        if not rows:
            return LLMResult(None, "", "end_turn", usage, 0.0, error="no candidates", model=self.model)
        if self.mode == "invalid":
            sid = "S99"
        elif self.mode == "first":
            sid = rows[0][0]
        else:
            sid = max(rows, key=lambda r: (r[1], -int(r[0][1:])))[0]
        name = schema.__name__
        if name == "SpecialistReport":
            obj = {"assessments": [{"shot_id": s, "score": int(round(p / 10)), "note": "fake"}
                                   for s, p in rows],
                   "top_choice": sid, "summary": f"fake {agent} view"}
        elif name == "CoordinatorDecision":
            obj = {"shot_id": sid, "intent": "offense", "rationale": "fake coordinator",
                   "confidence": 0.6, "overruled": []}
        elif name == "CriticVerdict":
            obj = {"approve": True, "reason": "fake critic", "alternative_shot_id": None}
        else:
            obj = {"shot_id": sid, "intent": "offense", "rationale": "fake single agent",
                   "confidence": 0.6}
        return LLMResult(schema.model_validate(obj), "", "end_turn", usage, 0.0, model=self.model)


def make_client(kind: str = "ollama", **kw) -> LLMClient:
    if kind == "fake":
        return FakeLLM(kw.get("mode", "best"))
    if kind == "ollama":
        return OllamaClient(**kw)
    return ClaudeVertexClient(**kw)
