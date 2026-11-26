"""Record the numerical libraries and available solver backends."""
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import scipy
import torch


def main():
    result = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'cuda_available': torch.cuda.is_available(),
        'numpy_version': numpy.__version__, 'scipy_version': scipy.__version__,
        'torch_version': torch.__version__,
        'highs_backend': 'scipy.optimize.milp / scipy.optimize.linprog(method=highs)',
        'gurobipy_installed': bool(importlib.util.find_spec('gurobipy')),
        'highspy_installed': bool(importlib.util.find_spec('highspy')),
    }
    output = Path('output/environment.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
