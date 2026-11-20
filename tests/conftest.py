import os

import pytest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


@pytest.fixture(scope="session", autouse=True)
def _warm_jit():
    """Compile pooltool's Numba kernels once per test session."""
    from cuemind.sim.execute import warm_up

    warm_up()
