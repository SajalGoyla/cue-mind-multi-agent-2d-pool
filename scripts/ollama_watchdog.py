"""Restart the local Ollama server when its runner grows too large in RAM (D-028).

During long runs Ollama's `llama-server` runner grew steadily in system RAM (3.7 -> 5.1 GB
in ~15 min while the model itself sat in VRAM) until the machine ran short of memory. The
LLM client retries infrastructure failures (D-027), so a restart is invisible to the
experiments apart from a short pause.

Used by scripts/run_queue.py (--watchdog-mb), or standalone:
    python scripts/ollama_watchdog.py --limit-mb 3500
"""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil

OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("CUEMIND_OLLAMA_MODEL", "qwen2.5:7b-instruct")


def runner_rss_mb() -> float:
    total = 0.0
    for p in psutil.process_iter(["name", "memory_info"]):
        if (p.info["name"] or "").lower().startswith("llama-server"):
            total += p.info["memory_info"].rss / 2**20
    return total


def restart_ollama(parallel: int = 2, log=print) -> None:
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() in ("ollama.exe", "ollama app.exe", "llama-server.exe",
                                              "ollama", "llama-server"):
            try:
                p.kill()
            except psutil.Error:
                pass
    time.sleep(3)
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(parallel), "OLLAMA_KEEP_ALIVE": "4h"}
    exe = str(OLLAMA_EXE) if OLLAMA_EXE.exists() else "ollama"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=flags)
    for _ in range(60):
        try:
            httpx.get(f"{HOST}/api/version", timeout=3)
            break
        except httpx.HTTPError:
            time.sleep(2)
    try:  # load the model so the next real request does not pay the load time
        httpx.post(f"{HOST}/api/generate", json={"model": MODEL, "prompt": "hi", "stream": False,
                                                  "options": {"num_ctx": 8192, "num_predict": 1}},
                   timeout=300)
    except httpx.HTTPError as exc:
        log(f"watchdog: warm-up failed: {exc!r}")


class Watchdog(threading.Thread):
    def __init__(self, limit_mb: float, period_s: float = 60, parallel: int = 2, log=print):
        super().__init__(daemon=True)
        self.limit_mb, self.period_s, self.parallel, self.log = limit_mb, period_s, parallel, log
        self.restarts = 0

    def run(self) -> None:
        while True:
            time.sleep(self.period_s)
            rss = runner_rss_mb()
            if rss > self.limit_mb:
                self.log(f"watchdog: llama-server RSS {rss:.0f} MB > {self.limit_mb:.0f} MB, restarting Ollama")
                restart_ollama(self.parallel, self.log)
                self.restarts += 1
                self.log(f"watchdog: restart {self.restarts} done (RSS now {runner_rss_mb():.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-mb", type=float, default=3500)
    ap.add_argument("--period-s", type=float, default=60)
    ap.add_argument("--restart-now", action="store_true")
    a = ap.parse_args()
    if a.restart_now:
        restart_ollama()
        print(f"restarted; runner RSS {runner_rss_mb():.0f} MB")
    else:
        w = Watchdog(a.limit_mb, a.period_s)
        w.start()
        w.join()
