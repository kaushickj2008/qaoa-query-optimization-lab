"""One JSON experiment per process; no shell commands or user code accepted."""

import json
import sys

from dashboard_service import run_experiment


if __name__ == "__main__":
    result = run_experiment(json.load(sys.stdin))
    print(json.dumps(result, default=lambda value: value.tolist(), allow_nan=False))
