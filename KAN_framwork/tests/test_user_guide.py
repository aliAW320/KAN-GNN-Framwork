"""Keep every self-contained Python example in the detailed guide executable."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

GUIDE = Path(__file__).parents[1] / "docs" / "USER_GUIDE.md"
EXAMPLES = re.findall(r"^```python\n(.*?)^```", GUIDE.read_text(encoding="utf-8"),
                      flags=re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize("index,source", list(enumerate(EXAMPLES, start=1)),
                         ids=[f"example-{i}" for i in range(1, len(EXAMPLES) + 1)])
def test_guide_example(index, source, tmp_path):
    # Optional extension examples require the separately installed package.
    if "from kan_transforms import" in source:
        pytest.importorskip("kan_transforms")
    script = tmp_path / f"guide_example_{index}.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The documentation checks have a CPU target even on GPU-equipped machines.
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Example {index}\n{result.stdout}\n{result.stderr}"
