"""Check a committed-revision wheel from outside the source checkout.

Usage: python scripts/check_wheel_imports.py path/to/package.whl
The interpreter must have the package dependencies installed.
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    wheel = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            assert "molecular_qm_turbomole/lib/parallel.py" in archive.namelist()
            archive.extractall(directory)
        code = """
import importlib
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
for name in (
    'molecular_qm_turbomole.lib.parallel',
    'molecular_qm_turbomole.nodes.turbomole2',
    'molecular_qm_turbomole.nodes.turbomole_ricc2',
    'hyperpolarizibility.hyperpol_runner',
    'hyperpolarizibility.hyperpolarizibility2',
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), module.__file__
    print('Imported', name)
"""
        subprocess.run([sys.executable, '-I', '-c', code, directory], cwd=directory, check=True)


if __name__ == '__main__':
    main()
