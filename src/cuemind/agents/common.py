"""Shared helpers for LLM policies: id validation with one corrective retry, usage totals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from cuemind.llm.client import LLMClient, LLMResult

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                "cost_usd")


def call_with_id_check(client: LLMClient, *, system: str, user: str, schema: type[BaseModel],
                       agent: str, valid_ids: set[str], repeat_idx: int = 0,
                       id_field: str = "shot_id") -> tuple[LLMResult, list[dict]]:
    """Call the model; if the returned id is not in the menu (or nothing parsed), retry once
    with a short corrective note. Returns the final result and a log of every attempt."""
    attempts: list[dict] = []
    res = client.complete(system=system, user=user, schema=schema, agent=agent,
                          repeat_idx=repeat_idx)
    attempts.append({"agent": agent, "attempt": 0, **res.log()})
    if _valid(res, valid_ids, id_field):
        return res, attempts
    got = getattr(res.parsed, id_field, None) if res.parsed is not None else None
    note = (f"\n\nYour previous answer used the id {got!r}, which is not in the list. "
            if got is not None else "\n\nYour previous answer could not be read. ")
    note += f"Valid ids are: {', '.join(sorted(valid_ids))}. Answer again."
    res2 = client.complete(system=system, user=user + note, schema=schema, agent=agent,
                           repeat_idx=repeat_idx)
    attempts.append({"agent": agent, "attempt": 1, **res2.log()})
    return res2, attempts


def _valid(res: LLMResult, valid_ids: set[str], id_field: str) -> bool:
    return res.parsed is not None and getattr(res.parsed, id_field, None) in valid_ids


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
