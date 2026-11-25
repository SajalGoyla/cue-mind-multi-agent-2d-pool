"""Shared helpers for LLM policies: id validation with one corrective retry, usage totals."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from cuemind.llm.client import LLMClient, LLMResult

USAGE_FIELDS = ("input_tokens", "output_tokens")


def call_with_check(client: LLMClient, *, system: str, user: str, schema: type[BaseModel],
                    agent: str, is_valid: Callable[[BaseModel], bool], valid_ids: set[str],
                    repeat_idx: int = 0) -> tuple[LLMResult, list[dict]]:
    """Call the model; if the answer is unusable (nothing parsed, or `is_valid` fails), retry
    once with a short note listing the valid ids. Returns the final result and every attempt."""
    attempts: list[dict] = []
    res = client.complete(system=system, user=user, schema=schema, agent=agent,
                          repeat_idx=repeat_idx)
    attempts.append({"agent": agent, "attempt": 0, **res.log()})
    if res.parsed is not None and is_valid(res.parsed):
        return res, attempts
    note = ("\n\nYour previous answer could not be used. Use only these ids: "
            f"{', '.join(sorted(valid_ids))}. Answer again.")
    res2 = client.complete(system=system, user=user + note, schema=schema, agent=agent,
                           repeat_idx=repeat_idx)
    attempts.append({"agent": agent, "attempt": 1, **res2.log()})
    return res2, attempts


def sum_usage(attempts: list[dict], wall_s: float) -> dict[str, Any]:
    tot: dict[str, Any] = {k: 0.0 for k in USAGE_FIELDS}
    tot["calls"] = 0
    tot["cached_calls"] = 0
    for a in attempts:
        u = a.get("usage") or {}
        for k in USAGE_FIELDS:
            tot[k] += float(u.get(k, 0.0))
        tot["calls"] += 1
        tot["cached_calls"] += int(bool(a.get("cached")))
    tot["latency_s"] = round(wall_s, 3)
    return tot
