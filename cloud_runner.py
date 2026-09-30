"""Bounded, subprocess-isolated execution for the shared Streamlit demo."""

import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

from dashboard_service import validate

PUBLIC_PROBLEMS = ("maxcut", "join3", "bushy4_valid", "deep4_valid")
RUN_LOCK = Lock()
TIMEOUT_SECONDS = 120


def validate_public(config: dict) -> dict:
    """Limit shared compute; the local dashboard retains the larger experiments."""
    config = validate(config)
    if config["problem"] not in PUBLIC_PROBLEMS:
        raise ValueError("The 10-qubit experiments are available locally only.")
    if config["p"] > 2 or config["maxiter"] > 100 or config["shots"] > 2000:
        raise ValueError("Demo limits: depth 2, budget 100, and 2,000 samples.")
    return config


def run_public(config: dict) -> dict:
    """Run one job per application process, killing the child on timeout.

    A nonblocking lock prevents queues from growing across browser sessions.
    This is a single-instance portfolio demo, not a distributed job service.
    """
    config = validate_public(config)
    if not RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another visitor is running an experiment. Please try again shortly.")
    try:
        env = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("cloud_worker.py"))],
            input=json.dumps(config), capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, check=True, env=env,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("The run exceeded two minutes. Try depth 1 or a smaller budget.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("The simulator could not finish. Try a smaller experiment.") from error
    finally:
        RUN_LOCK.release()
