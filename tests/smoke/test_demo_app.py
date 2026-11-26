"""Demo smoke test: the app loads, renders a position, and runs both systems offline."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "demo" / "app.py")


@pytest.mark.slow
def test_demo_app_runs_offline():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    assert at.radio[0].value == "Offline (fake LLM)"
    run = next(b for b in at.button if "Run single agent" in b.label)
    run.click().run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Choice: S" in text and "Single agent" in text and "specialist" in text
    assert "Stat" in text
