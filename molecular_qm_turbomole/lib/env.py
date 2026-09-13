import os
from pathlib import Path


def prepend_tm_env(run_script: str) -> str:
    """Prepend a POSIX-shell prelude that loads a Turbomole environment script."""
    here = Path(__file__).resolve().parent
    default_env = str((here / "scripts" / "turbomole_int_nano.sh").resolve())
    home_env = os.path.expandvars("$HOME/simstack/turbomole_int_nano.sh")

    prelude = f"""set -e
if [ "${{TM_DEBUG:-0}}" = "1" ]; then
  set -x
fi

TM_ENV_SCRIPT="${{TURBOMOLE_ENV_SCRIPT:-}}"
if [ -z "$TM_ENV_SCRIPT" ]; then
  if [ -f "{default_env}" ]; then
    TM_ENV_SCRIPT="{default_env}"
  elif [ -f "{home_env}" ]; then
    TM_ENV_SCRIPT="{home_env}"
  fi
fi

if [ -n "$TM_ENV_SCRIPT" ] && [ -f "$TM_ENV_SCRIPT" ]; then
  echo "[TM ENV] sourcing $TM_ENV_SCRIPT"
  . "$TM_ENV_SCRIPT"
fi

set -e
set +u 2>/dev/null || true

if ! command -v define >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
      . /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
      . /usr/share/Modules/init/bash
    elif [ -f /usr/share/module/init/bash ]; then
      . /usr/share/module/init/bash
    fi
  fi
  if command -v module >/dev/null 2>&1; then
    module load turbomole/7.6 || true
  fi
fi

command -v define >/dev/null 2>&1 || {{
  echo "[TM ENV ERROR] define not found in PATH. Set TURBOMOLE_ENV_SCRIPT or fix modules/env script."
  exit 127
}}
"""
    return prelude + "\n" + run_script


def _aoforce_script_lines() -> list[str]:
    return [
        "aoforce > aoforce.out 2>&1 || { "
        'echo "[TM ERROR] aoforce failed."; '
        "tail -n 200 aoforce.out || true; "
        "exit 24; "
        "}",
        "test -f vibspectrum || { "
        'echo "[TM ERROR] vibspectrum not produced."; '
        "ls -la; "
        "tail -n 80 aoforce.out || true; "
        "exit 25; "
        "}",
    ]


def build_frequency_script() -> str:
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        'echo "[TM] aoforce: $(command -v aoforce)"',
        *_aoforce_script_lines(),
        'echo "[TM] produced files:"',
        "ls -la",
    ]
    return "\n".join(lines)


def build_hyperpolarizability_script() -> str:
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        'echo "[TM] escf: $(command -v escf)"',
        "escf > escf.out 2>&1 || { "
        'echo "[TM ERROR] escf failed for hyperpol."; '
        "tail -n 200 escf.out || true; "
        "exit 3; "
        "}",
        "test -f hyperpols || { "
        'echo "[TM ERROR] hyperpols file not produced."; '
        "ls -la; "
        "tail -n 200 escf.out || true; "
        "exit 4; "
        "}",
        'echo "[TM] produced files:"',
        "ls -la",
    ]
    return "\n".join(lines)
